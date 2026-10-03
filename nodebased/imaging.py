"""CPU reference evaluator. Internal RGBA is float32, scene-linear, premultiplied."""
from __future__ import annotations

from collections import OrderedDict
import copy
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import threading
import time

import numpy as np
from PySide6.QtGui import QImage, QImageReader

from . import cachetier
from . import groups
from . import metadata
from . import shapes
from . import tiers
from . import tracker
from .raster import Raster, as_array, scale_window
from .tiers import Region


from .cancellation import Cancelled


def _whitewater_cache_state(value):
    from .simcache import State
    return State({"position": value.positions, "velocity": value.velocities, "size": value.sizes,
                  "age": value.ages, "life": value.lifetimes, "id": value.ids,
                  "kind": value.kinds}, {"next_id": int(value.next_id)}, copy=False)


def _whitewater_state(value):
    from .whitewater import WhitewaterState
    a = value.arrays
    return WhitewaterState(a["position"], a["velocity"], a["size"], a["age"], a["life"], a["id"],
                           a["kind"], int(value.meta.get("next_id", 0)))


def _volume_settings(params):
    """The Render3D smoke knobs as a `volumerender.VolumeSettings`."""
    from .volumerender import VolumeSettings
    return VolumeSettings(
        params["volume_step_size"], params["volume_density_scale"],
        params["volume_shadow_density"], int(params["volume_shadow_steps"]),
        params["volume_scattering"], params["volume_absorption"],
        (params["volume_red"], params["volume_green"], params["volume_blue"]),
        params["volume_fps"], params["volume_depth_threshold"],
        params.get("volume_motion_blur", 0.0), int(params.get("volume_motion_samples", 8)),
        params.get("volume_anisotropy", 0.0), params.get("volume_multi_scatter", 0.0),
        params.get("volume_multi_scatter_blur", 0.5), params.get("volume_fire_intensity", 0.0),
        params.get("volume_temperature_scale", 1500.0), params.get("volume_fire_threshold", 600.0),
        params.get("volume_fire_light", 1.0), str(params.get("volume_fire_ramp", "")),
        params.get("volume_quality", "custom"))


def _path_settings(params, mode):
    """The path tracer's `PathSettings` for a Render3D in `pathtrace` mode, else None."""
    if mode != "pathtrace":
        return None
    from . import pathtrace
    return pathtrace.settings_from_params(params)


def _add_cryptomatte(value, scene, camera, params, cancel, mode):
    """`value` (a `Raster`) with the Cryptomatte layers and header entries added, when the node's
    `cryptomatte` knob is on. Off leaves `value` untouched, so an old document renders the same EXR
    byte for byte. The pass always runs on the CPU reference renderer (`scene3d.render`), whichever
    backend the beauty pass used: a GPU coverage buffer is not available yet (docs/PARITY_2D.md
    Keyer row 8, Render3D's known limit)."""
    if not params.get("cryptomatte"):
        return value
    mode = "raytrace" if mode == "pathtrace" else mode      # coverage wants the antialiased CPU visibility, not noise
    from . import cryptomatte3d
    from .raster import Raster as _Raster
    layers, metadata = cryptomatte3d.render_cryptomatte(
        scene, camera, params["width"], params["height"], samples=params["samples"],
        levels=int(params.get("cryptomatte_levels", 6)), mode=mode, cancel=cancel)
    merged_layers = {**(value.layers or {}), **{name: _Raster.of(arr) for name, arr in layers.items()}}
    merged_meta = {**(value.meta or {}), **cryptomatte3d.cryptomatte_header(metadata)}
    return _Raster(value.pixels, value.data, value.display, merged_layers, merged_meta)


_LABEL_GLYPHS = {
    "A":("010","101","111","101","101"), "B":("110","101","110","101","110"),
    "C":("011","100","100","100","011"), "D":("110","101","101","101","110"),
    "E":("111","100","110","100","111"), "F":("111","100","110","100","100"),
    "G":("011","100","101","101","011"), "H":("101","101","111","101","101"),
    "I":("111","010","010","010","111"), "J":("001","001","001","101","010"),
    "K":("101","101","110","101","101"), "L":("100","100","100","100","111"),
    "M":("101","111","111","101","101"), "N":("101","111","111","111","101"),
    "O":("010","101","101","101","010"), "P":("110","101","110","100","100"),
    "Q":("010","101","101","111","011"), "R":("110","101","110","101","101"),
    "S":("011","100","010","001","110"), "T":("111","010","010","010","010"),
    "U":("101","101","101","101","111"), "V":("101","101","101","101","010"),
    "W":("101","101","111","111","101"), "X":("101","101","010","101","101"),
    "Y":("101","101","010","010","010"), "Z":("111","001","010","100","111"),
    "0":("111","101","101","101","111"), "1":("010","110","010","010","111"),
    "2":("110","001","010","100","111"), "3":("110","001","010","001","110"),
    "4":("101","101","111","001","001"), "5":("111","100","110","001","110"),
    "6":("011","100","110","101","010"), "7":("111","001","010","010","010"),
    "8":("010","101","010","101","010"), "9":("010","101","011","001","110"),
    "-":("000","000","111","000","000"), "_":("000","000","000","000","111"),
    ".":("000","000","000","000","010"), " ":("000","000","000","000","000"),
}


def _paint_contact_label(pixels, text, x, y, scale):
    """Draw a tiny dependency-free 3x5 label in white, preserving HDR under the glyphs."""
    h, w = pixels.shape[:2]
    cursor = int(x)
    for char in str(text).upper():
        glyph = _LABEL_GLYPHS.get(char, _LABEL_GLYPHS["_"])
        for gy, line in enumerate(glyph):
            for gx, bit in enumerate(line):
                if bit != "1":
                    continue
                x0, y0 = cursor + gx*scale, int(y) + gy*scale
                xa, ya = min(w, x0+scale), min(h, y0+scale)
                if x0 < 0 or y0 < 0 or xa <= 0 or ya <= 0:
                    continue
                pixels[max(0,y0):ya, max(0,x0):xa, 3] = 1.0
                pixels[max(0,y0):ya, max(0,x0):xa, :3] = 1.0
        cursor += 4*scale
        if cursor >= w:
            break


def curve_tool_metrics(pixels, box=(0, 0, 0, 0), previous_luminance=None,
                       autocrop_mode="alpha", autocrop_color=(0.0, 0.0, 0.0), autocrop_tolerance=0.0):
    """Average RGBA in a display-space box, autocrop bounds, brightest/dimmest pixel, exposure step.

    Mirrors Nuke's CurveTool "Curve Type" analyses (reference guide): Avg Intensities (the
    per-channel averages), AutoCrop (`autocrop_mode` "alpha" bounds pixels with any coverage,
    same as before; "color" bounds pixels that are NOT within `autocrop_tolerance` of
    `autocrop_color` by the largest per-channel difference, Nuke's chosen-colour-plus-tolerance
    crop), Max Luma Pixel (which the reference guide states also reports the dimmest pixel, so
    both extremes and their values are returned), and Exposure Difference (the change in average
    luminance from the previous sampled frame, zero on the first frame of a range or when the
    caller has no earlier sample to compare).
    """
    pixels = np.asarray(pixels, dtype=np.float32)
    x, y, width, height = map(int, box)
    x0, y0 = max(0, x), max(0, y)
    x1 = pixels.shape[1] if width <= 0 else min(pixels.shape[1], x + width)
    y1 = pixels.shape[0] if height <= 0 else min(pixels.shape[0], y + height)
    region = pixels[y0:y1, x0:x1]
    if not region.size:
        raise ValueError("Analysis box does not overlap the image")
    if autocrop_mode == "color":
        target = np.asarray(autocrop_color, dtype=np.float32)
        diff = np.max(np.abs(pixels[..., :3] - target), axis=2)
        ys, xs = np.nonzero(diff > float(autocrop_tolerance))
    else:
        ys, xs = np.nonzero(pixels[..., 3] > 1e-6)
    crop = ((float(xs.min()), float(ys.min()), float(xs.max()-xs.min()+1), float(ys.max()-ys.min()+1))
            if len(xs) else (0.0, 0.0, 0.0, 0.0))
    luminance = pixels[..., :3].mean(axis=2)
    max_y, max_x = np.unravel_index(int(np.argmax(luminance)), luminance.shape)
    min_y, min_x = np.unravel_index(int(np.argmin(luminance)), luminance.shape)
    average_luminance = float((region[..., 0].mean() + region[..., 1].mean() + region[..., 2].mean()) / 3.0)
    exposure_diff = 0.0 if previous_luminance is None else average_luminance - float(previous_luminance)
    return {"average_r": float(region[..., 0].mean()), "average_g": float(region[..., 1].mean()),
            "average_b": float(region[..., 2].mean()), "average_a": float(region[..., 3].mean()),
            "average_luminance": average_luminance,
            "crop_x": crop[0], "crop_y": crop[1], "crop_width": crop[2], "crop_height": crop[3],
            "max_x": float(max_x), "max_y": float(max_y), "max_value": float(luminance[max_y, max_x]),
            "min_x": float(min_x), "min_y": float(min_y), "min_value": float(luminance[min_y, min_x]),
            "exposure_diff": exposure_diff}


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
    source_frame = int(math.floor((frame if frame is not None else 0) + int(frame_offset) + 0.5))
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
    raster = read_media_raster(resolved, colorspace, alpha_mode, layer, subimage)
    # Where the pixels came from, in Nuke's key names. Every value follows from the resolved file
    # (input/frame only when that file is the one the sequence names for this frame), so the Read's
    # cache key, which fingerprints the resolved file, already covers the metadata.
    from .media import sequence_path
    meta = {"input/filename": str(resolved), "input/width": str(raster.display.width),
            "input/height": str(raster.display.height)}
    if str(resolved) == sequence_path(path, source_frame) and str(resolved) != str(path):
        meta["input/frame"] = str(source_frame)
    raster.meta = {**meta, **(raster.meta or {})}
    return raster


def read_image_region(path, region, colorspace="Auto", alpha_mode="Auto", layer="", subimage=0,
                      frame_offset=0, missing="error", frame=None):
    """Acquire a bounded Read region at full resolution.

    The returned Raster is anchored at ``region`` even when it extends into EXR overscan or
    beyond a source data window. This is the source-side counterpart to TileExecutor's bounded
    compose API: a viewport request no longer needs a full image decode merely to slice it.
    """
    from .media import nearest_sequence_path, read_media_region, read_media_raster, resolve_source_path
    source_frame = int(math.floor((frame if frame is not None else 0) + int(frame_offset) + 0.5))
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
    source_frame = int(math.floor((frame if frame is not None else 0) + int(frame_offset) + 0.5))
    resolved, exists = resolve_source_path(path, source_frame, missing)
    if resolved is None and not exists:
        resolved = nearest_sequence_path(path, source_frame)
        if resolved is None:
            raise ValueError(f'No frames found for sequence {path}')
    return read_media_bounds(resolved, subimage)


# The clipping warning ("zebra"): a pixel whose adjusted scene-linear value is above ZEBRA_HIGH is
# striped red, one below ZEBRA_LOW striped blue. The thresholds are shown next to the toggle.
ZEBRA_HIGH = 1.0
ZEBRA_LOW = 0.0
ZEBRA_STRIPE = 6


def zebra_masks(rgb, height, width):
    """(over, under) boolean masks: the clipped pixels that fall on a stripe of the pattern."""
    yy, xx = np.ogrid[:height, :width]
    stripe = ((xx + yy) // ZEBRA_STRIPE) % 2 == 0
    return (rgb > ZEBRA_HIGH).any(axis=2) & stripe, (rgb < ZEBRA_LOW).any(axis=2) & stripe


def to_qimage(frame, exposure=0.0, channel="RGB", background="black", view="sRGB", look=None):
    """Compose a display image over `background` ("black" or "checker").

    Frames are premultiplied, so black is a true no-op: transparent regions stay at
    zero and a partially transparent edge keeps the value the graph produced. The
    checkerboard reads alpha at a glance but tints every pixel it shows through,
    which is why it is no longer the default.

    `exposure` is the viewer gain in f-stops. `look` carries the other display-only viewer
    controls, `{"gamma": float, "zebra": bool}`: gamma is applied after gain and before the view
    transform (`v ** (1 / gamma)`, sign kept), and the clipping warning tests that same adjusted
    scene-linear value and paints over the displayable buffer only.
    """
    alpha = frame[..., 3:4]
    if channel == "A":
        rgb = np.repeat(alpha, 3, axis=2)
    else:
        rgb = frame[..., :3] * (2.0 ** exposure)
        if channel in ("R", "G", "B"):
            rgb = np.repeat(rgb[..., "RGB".index(channel):"RGB".index(channel) + 1], 3, axis=2)
        from .color import display_rgb
        # The view transform is nonlinear, so it has to see straight (unassociated) colour;
        # feeding it premultiplied RGB bends every partially transparent pixel. `write_png`
        # has always unpremultiplied first, so the viewer and the exporter used to disagree
        # on exactly those pixels. Same order here: divide out alpha, transform, re-associate.
        weight = np.clip(alpha, 0, 1)
        straight = np.divide(rgb, weight, out=np.zeros_like(rgb), where=weight > 1e-8)
        gamma = float((look or {}).get("gamma", 1.0))
        if gamma != 1.0:
            straight = np.sign(straight) * np.abs(straight) ** np.float32(1.0 / gamma)
        rgb = display_rgb(straight, view) * weight
        zebra = (zebra_masks(straight, *straight.shape[:2]) if (look or {}).get("zebra") else None)
        if background == "checker":
            # Composited after the transform, so the checker keeps the tone it was authored
            # with under any view instead of being pushed through the display curve.
            yy, xx = np.ogrid[:frame.shape[0], :frame.shape[1]]
            levels = display_rgb(np.array([[[0.055] * 3, [0.095] * 3]], np.float32), view)[0, :, 0]
            bg = np.where((xx // 16 + yy // 16) % 2 == 0, levels[0], levels[1]).astype(np.float32)
            rgb = rgb + bg[..., None] * (1 - weight)
    rgb8 = (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)
    if channel != "A" and (look or {}).get("zebra"):
        over, under = zebra
        rgb8[over] = (255, 0, 0)
        rgb8[under] = (0, 90, 255)
    return QImage(rgb8.data, rgb8.shape[1], rgb8.shape[0], rgb8.strides[0], QImage.Format.Format_RGB888).copy()


def write_png(path, frame):
    if Path(path).suffix.lower() != ".png":
        raise ValueError("Export path must end in .png")
    alpha = np.clip(frame[..., 3:4], 0, 1)
    straight = np.divide(frame[..., :3], alpha, out=np.zeros_like(frame[..., :3]), where=alpha > 1e-8)
    # Through the OCIO sRGB view, not a bare transfer curve: the working space is ACEScg, so
    # encoding needs the AP1 -> Rec.709 gamut conversion as well as the OETF. A hardcoded
    # `linear_to_srgb` here would silently desaturate every export.
    from .color import display_rgb
    rgba = np.concatenate((np.clip(display_rgb(straight, "sRGB"), 0, 1), alpha), axis=2)
    rgba8 = (rgba * 255 + 0.5).astype(np.uint8)
    image = QImage(rgba8.data, rgba8.shape[1], rgba8.shape[0], rgba8.strides[0], QImage.Format.Format_RGBA8888).copy()
    # QSaveFile preserves an existing export on write failure.
    from PySide6.QtCore import QSaveFile, QIODevice
    target = QSaveFile(str(path))
    if not target.open(QIODevice.OpenModeFlag.WriteOnly):
        raise ValueError(target.errorString())
    if not image.save(target, "PNG") or not target.commit():
        raise ValueError(f"Cannot export PNG: {target.errorString()}")


# Producers with their own time mapping (TIME_MODEL.md's "clip" shape): each remaps the timeline
# frame it is evaluated at onto a different frame for its single required input, via a nested
# `Evaluator.evaluate_raster` call rather than by reusing this walk's `values[source]`.
# Kinds whose result depends on the timeline frame through metadata expressions or a timecode.
_FRAME_METADATA_KINDS = ("ModifyMetaData", "AddTimeCode", "BurnIn")


def _inherited_metadata(kind, images):
    """The metadata a node's result carries when the kernel set none: the main input's, which for the
    two-input Merge family is B (the background pipe), as in Nuke."""
    from .core import MERGE_LIKE_KINDS
    order = images[1::-1] if kind in MERGE_LIKE_KINDS else images
    # Inputs are not always rasters (Relight takes a Camera, 3D nodes take scenes): only rasters carry meta.
    return next((m for m in (getattr(image, "meta", None) for image in order) if m is not None), None)


_TIME_REMAP_KINDS = ("TimeOffset", "FrameHold", "Retime", "TimeClip", "FrameRange", "AppendClip")

# M1 gate, "interactive cancellation" (docs/M1_GATE.md): `evaluate_raster` only checks `cancel`
# once per node, so a single node whose own kernel exceeds the 100 ms budget cannot be
# interrupted inside itself. Measured at 4K: Grade ~140 ms, Saturation ~160 ms, ColorCorrect
# ~290 ms -- all already over budget on their own. Each of the three is a pure elementwise
# (per-pixel-independent) kernel with an identity region-of-interest rule (`tiers._identity`:
# halo (0, 0), already proven pixel-identical on the tile path's own 256px tiles by the golden
# image suite), so it can be split into row bands of the same edge the tile executor uses and
# checked between bands without changing a single output pixel.
_ROW_CHUNKED_MASK_MIX_KINDS = frozenset({"Grade", "ColorCorrect", "Saturation"})
_CANCEL_CHUNK_ROWS = 256


def _range_frame(frame, first, last, before, after):
    """Map ``frame`` onto ``[first, last]`` under the before/after policy, returning
    ``(frame_in_range, black)``. hold clamps; loop repeats the range (period ``last - first + 1``);
    bounce ping-pongs without repeating the end frames (period ``2 * (last - first)``, so 1..4 runs
    1 2 3 4 3 2 1 2 ...); black returns the nearest in-range frame (so the size is known) with
    ``black`` set, and the caller blanks it."""
    lo, hi = (first, last) if first <= last else (last, first)
    if lo <= frame <= hi:
        return frame, False
    policy = before if frame < lo else after
    nearest = lo if frame < lo else hi
    if policy == "black":
        return nearest, True
    if policy == "loop":
        return lo + (frame - lo) % (hi - lo + 1), False
    if policy == "bounce":
        if hi == lo:
            return lo, False
        period = 2 * (hi - lo)
        r = (frame - lo) % period
        return lo + (r if r <= hi - lo else period - r), False
    return nearest, False   # hold


def _time_clip_frame(kind, params, frame):
    """``(source_frame, black)`` for TimeClip and FrameRange. TimeClip first maps the timeline
    frame by ``frame - time_offset`` (TimeOffset's sign) and, unless ``frame_range_type`` is
    "all", applies ``[first, last]`` with ``before``/``after``. FrameRange applies
    ``[first_frame, last_frame]`` to the frame directly."""
    frame = int(frame)
    if kind == "TimeClip":
        mapped = frame - int(params.get("time_offset", 0))
        if params.get("frame_range_type", "custom") == "all":
            return mapped, False
        first, last = int(params.get("first", 1)), int(params.get("last", 100))
    else:
        mapped = frame
        first, last = int(params.get("first_frame", 1)), int(params.get("last_frame", 100))
    return _range_frame(mapped, first, last, params.get("before", "hold"), params.get("after", "hold"))


def _branch_frame_range(nodes, key):
    """The frame range a clip's branch presents downstream, or None when unknown. Known only when
    a FrameRange or a custom-range TimeClip is reached from `key` through bypassed nodes and Dots
    (the document has no other notion of a branch's range); a TimeClip presents its range shifted
    by its offset, since that is the frame span over which it maps into [first, last]."""
    from .core import SPECS, bypass_slot
    for _ in range(len(nodes) + 1):
        node = nodes[key]
        kind = node["type"]
        if node["disabled"] or kind in ("Dot", "NoOp", "PostageStamp", "Output"):
            slot = bypass_slot(node)
            key = None if slot is None else node["inputs"].get(slot)
            if key is None:
                return None
            continue
        params = node["params"]
        if kind == "FrameRange":
            first, last = int(params["first_frame"]), int(params["last_frame"])
            return (min(first, last), max(first, last))
        if kind == "TimeClip" and params.get("frame_range_type", "custom") == "custom":
            shift = int(params.get("time_offset", 0))
            first, last = int(params["first"]) + shift, int(params["last"]) + shift
            return (min(first, last), max(first, last))
        return None
    return None


def _append_clip_plan(doc_nodes, node, params, frame):
    """The nested evaluations AppendClip needs at timeline ``frame``: a list of
    ``(slot, source_frame, weight)`` with weights summing to 1 (one entry, or two inside a
    dissolve).

    Wired clips play head to tail from ``first_frame``. A clip's length is its branch's frame range
    when directly known (`_branch_frame_range`), and it is sampled from that range's first frame;
    otherwise its length is the ``length<i>`` knob (0 skips the clip) and it is sampled from frame 1.
    ``dissolve`` frames overlap each clip's tail with the next clip's head (clamped to one frame
    less than either clip); inside the overlap the outgoing clip is weighted ``1 - w`` and the
    incoming ``w = (k + 1) / (dissolve + 1)`` at overlap index ``k``. Before the first clip and
    after the last the first frame / last frame holds. If more than two clips overlap on one frame
    (dissolve longer than a middle clip), the first two are blended and the rest ignored."""
    from .core import SPECS
    clips = []
    for index, slot in enumerate(SPECS["AppendClip"]["optional_inputs"]):
        source = node["inputs"].get(slot)
        if source is None:
            continue
        span = _branch_frame_range(doc_nodes, source)
        if span is not None:
            length, start = span[1] - span[0] + 1, span[0]
        else:
            length, start = int(params.get(f"length{index}", 100)), 1
        if length > 0:
            clips.append((slot, length, start))
    if not clips:
        raise ValueError(f"{node['name']}: connect at least one clip input")
    dissolve = int(params.get("dissolve", 0))
    starts, overlaps, cursor = [], [], int(params.get("first_frame", 1))
    for i, (_, length, _) in enumerate(clips):
        starts.append(cursor)
        overlap = 0
        if i + 1 < len(clips):
            overlap = max(0, min(dissolve, length - 1, clips[i + 1][1] - 1))
        overlaps.append(overlap)
        cursor += length - overlap
    frame = int(frame)
    live = [i for i, (_, length, _) in enumerate(clips) if starts[i] <= frame < starts[i] + length]
    if not live:
        i = 0 if frame < starts[0] else len(clips) - 1
        slot, length, first = clips[i]
        return [(slot, first if i == 0 else first + length - 1, 1.0)]
    if len(live) >= 2:
        a, b = live[0], live[1]
        k = frame - starts[b]
        weight = (k + 1) / (overlaps[a] + 1)
        (slot_a, _, first_a), (slot_b, _, first_b) = clips[a], clips[b]
        return [(slot_a, first_a + frame - starts[a], 1.0 - weight),
                (slot_b, first_b + frame - starts[b], weight)]
    i = live[0]
    slot, _, first = clips[i]
    return [(slot, first + frame - starts[i], 1.0)]


def _time_remap_frame(kind, params, frame):
    """The frame a time-remapping node's input is evaluated at, given the timeline `frame` the
    node itself is being evaluated at.

    TimeOffset: the input is evaluated at ``frame - time_offset``; ``reverse`` flips which
    direction the offset is applied (``frame + time_offset``), reversing the shift rather than the
    footage itself.

    FrameHold: holds on ``first_frame`` when ``increment`` is 0 (Nuke's own default). A positive
    increment advances the hold in steps: the input is evaluated at the nearest
    ``first_frame + k * increment`` at or before ``frame`` (``k`` clamped to 0 for a `frame` before
    ``first_frame`` — there is no earlier valid hold point to reach for).

    Retime (simplified per docs/PARITY_2D.md — nearest-frame sampling, no frame blending): a frame
    maps linearly from the output range onto the input range, anchored at each range's start and
    scaled by ``speed``: ``input_range_start + (frame - output_range_start) * speed``. The range
    end knobs are carried for Nuke-parity naming/duration bookkeeping; the simplified mapping does
    not consult them.
    """
    frame = int(frame)
    if kind == "TimeOffset":
        offset = int(params.get("time_offset", 0))
        return frame + offset if params.get("reverse") else frame - offset
    if kind == "FrameHold":
        first = int(params.get("first_frame", 1))
        increment = int(params.get("increment", 0))
        if increment <= 0 or frame <= first:
            return first
        return first + ((frame - first) // increment) * increment
    if kind == "Retime":
        input_start = float(params.get("input_range_start", 0))
        output_start = float(params.get("output_range_start", 0))
        speed = float(params.get("speed", 1.0))
        return int(round(input_start + (frame - output_start) * speed))
    if kind in ("TimeClip", "FrameRange"):
        return _time_clip_frame(kind, params, frame)[0]   # the "black" flag lives on that function
    raise ValueError(f"{kind} is not a single-input time-remapping kind")


def _cubic_taps(fraction):
    """The four distances a Catmull-Rom weight is taken at, for taps at -1, 0, +1 and +2 pixels."""
    return fraction + 1, fraction, fraction - 1, fraction - 2


LIGHT_MIXER_SLOTS = 8    # LightMixer: light groups with their own gain and colour knobs

class Evaluator:
    """Retained-result evaluator with a memory tier over an optional disk tier.

    The budget is sized for the machine rather than fixed (see `nodebased/cachetier.py`), because
    the previous fixed 256 MiB could not hold a single 8K result and evicted its own upstream at
    every node of a 4K chain — measured as zero cache hits and warm time-to-first-pixel matching
    cold to within 0.2%.
    """

    def __init__(self, cache_bytes=None, disk=None, sim=None, shared_budget=None, shared_name="raster"):
        # The disk tier is opt-in at construction rather than on by default: a library evaluator
        # must not start writing to a user's cache directory as a side effect of being imported.
        # The desktop app and the agent CLI pass `DiskCache.shared()` explicitly.
        self._fixed_budget = cachetier.default_memory_bytes() if cache_bytes is None else int(cache_bytes)
        self._shared_budget = shared_budget
        self._shared_name = shared_name
        if shared_budget is not None:
            shared_budget.register(shared_name, self)
        self.disk = cachetier.DiskCache(enabled=False) if disk is None else disk
        self.cache = OrderedDict()
        self.bytes = 0
        self.hits = 0
        self.misses = 0
        self.disk_hits = 0
        # Row bands `_run_row_chunked` has completed, across every node this instance has
        # evaluated. Real progress a caller (or a test synchronising with an in-flight
        # cancellation) can wait on, the same way it already waits on `misses`/`hits`.
        self.row_chunks = 0
        # Simulation frames (docs/SIMULATION.md). A ParticleEmitter3D keeps its frames in memory only,
        # so scrubbing within a session never re-solves. ParticleCache3D adds a persistent tier
        # when a `simcache.SimCache` is passed as `sim` (the app passes `SimCache.shared()`, like
        # the disk tier above); its budgets are per node, one store per distinct pair of budgets.
        from . import simcache
        self.sim_template = sim
        self._sim_memory = simcache.SimCache(enabled=False)
        self._sim_stores = {}
        self._rigid_solvers = {}
        self._liquid_feedback_versions = {}
        # The desktop app sets this callback to show ETA and stop CPU splat budget refusals.
        self.progress = None
        # Profile (Nuke's in-graph performance probe): one entry per evaluation of a Profile
        # node, keyed by node id. Session-only, like `self.hits`/`self.misses` -- not part of the
        # document, so it is never saved or loaded with the project.
        self.profile_log = {}

    @staticmethod
    def _environment_map(image):
        """The scene-linear straight RGB (H, W, 3) of an image wired to an Environment light."""
        pixels = np.asarray(image.to_display() if hasattr(image, "to_display") else image, dtype=np.float32)
        alpha = pixels[..., 3:4]
        return np.where(alpha > 1e-6, pixels[..., :3] / np.maximum(alpha, 1e-6), pixels[..., :3]).astype(np.float32)

    def _delit_cloud(self, cloud, params, cancel):
        """`cloud` with its intrinsic layer when the node's Delight is on (fitted once per cloud and
        settings, then served from the simcache); otherwise the cloud itself, untouched."""
        if params.get("splat_delight", "off") != "on":
            return cloud
        from . import intrinsics, splats
        identity = [*splats.fingerprint(params["splat_path"]), params["splat_orientation"], params["splat_colorspace"]]
        progress = self.progress
        report = None if progress is None else (lambda fraction: progress("delight", fraction, {}))
        layer = intrinsics.decompose_cached(
            cloud, identity, iterations=int(params.get("splat_delight_iterations", 12)),
            smoothness=float(params.get("splat_delight_smoothness", 0.5)),
            light_order=int(params.get("splat_delight_light_order", 1)),
            store=self.sim_store(256, 2048), cancel=cancel, progress=report)
        return intrinsics.attach(cloud, layer)

    def sim_store(self, memory_mb, disk_mb):
        """The ParticleCache3D store for one pair of budgets (megabytes)."""
        from . import simcache
        key = (int(memory_mb), int(disk_mb))
        store = self._sim_stores.get(key)
        if store is None:
            template = self.sim_template
            enabled = bool(template is not None and template.enabled and template.root is not None and key[1] > 0)
            store = simcache.SimCache(root=template.root if enabled else None,
                                      memory_budget=key[0] << 20, disk_budget=key[1] << 20, enabled=enabled)
            self._sim_stores[key] = store
        return store

    def clear(self):
        """Drop retained results from memory.

        The disk tier survives on purpose: it is what makes reopening a project cheap, and clearing
        it is a separate, explicit act.
        """
        self.cache.clear()
        self.bytes = 0

    _PROFILE_LOG_LIMIT = 500

    def _record_profile(self, node_id, frame, wall_time_ms, cache_hits):
        """Append one Profile row, keeping at most `_PROFILE_LOG_LIMIT` per node (oldest dropped)."""
        rows = self.profile_log.setdefault(node_id, [])
        rows.append({"frame": frame, "wall_time_ms": wall_time_ms, "cache_hits": cache_hits})
        del rows[:-self._PROFILE_LOG_LIMIT]

    @property
    def budget(self) -> int:
        if self._shared_budget is not None:
            return self._shared_budget.ceiling_for(self._shared_name)
        return self._fixed_budget

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
            if old.layers is None:   # the disk tier stores one array per raster; layers would be lost
                self.disk.put_raster(old_digest, old)
        self.cache[digest] = raster
        self.bytes += raster.nbytes

    def _motion_inputs(self, doc, node, params, frame, tier, cancel, base):
        """Render3D's scene and camera at the times motion blur and the `motion` pass need, and the cache fingerprint.

        Returns `(moments, later, fingerprint)`. `moments` is a list of `(scene, camera)` across the shutter when the
        node blurs, else None; `later` is `(scene, camera)` one frame on for the `motion` output and pass, else None.
        Each time is a nested evaluation of the scene and camera inputs at that (fractional) frame, so animated
        transforms, cameras and time-sampled meshes arrive through the graph; what a solver cached carries a velocity
        and is moved for the part of a frame past the solved one (nodebased/motionblur.py)."""
        from . import motionblur, scene3d
        output = params.get("render_output", "rgba")
        wants_motion = output == "motion" or (
            output == "multichannel" and "motion" in scene3d.parse_passes(params.get("passes", scene3d.DEFAULT_PASSES)))
        blur = bool(params.get("motion_blur", 0)) and float(params.get("shutter", 0.0)) > 0 and output != "motion"
        if not (blur or wants_motion):
            return None, None, None
        scene_key, camera_key = node["inputs"]["scene"], node["inputs"]["camera"]

        def at(time):
            scene, scene_digest = self.evaluate_raster(doc, scene_key, cancel=cancel, frame=time, tier=tier,
                                                       typed=True, return_digest=True)
            camera, camera_digest = self.evaluate_raster(doc, camera_key, cancel=cancel, frame=time, tier=tier,
                                                         typed=True, return_digest=True)
            return scene, camera, [scene_digest, camera_digest]
        moments, later, prints = None, None, []
        if blur:
            low, high = motionblur.shutter_window(frame, params["shutter"], params["shutter_offset"],
                                                  params["custom_offset"])
            moments = []
            for time in motionblur.shutter_times(low, high, params["motion_samples"]):
                scene, camera, digests = at(time)
                if isinstance(scene, scene3d.Scene):
                    scene = motionblur.advect_scene(scene, motionblur.solved_offset(time))
                moments.append((scene, camera))
                prints.append([round(time, 9), *digests])
        if wants_motion:
            base_scene, base_camera = base
            scene_later, camera_later, digests = at(frame + 1)
            later = (motionblur.next_scene(base_scene, scene_later) if isinstance(base_scene, scene3d.Scene) and
                     isinstance(scene_later, scene3d.Scene) else base_scene, camera_later)
            prints.append(["later", *digests])
        return moments, later, ["motion", *prints]

    def evaluate(self, doc, target=None, cancel: threading.Event | None = None, frame=None, tier=1):
        """Evaluate `target` and return its **display window** as an array.

        Overscan is discarded here and only here — see `evaluate_raster` when the caller needs the
        node's real data window. Keeping this signature returning a frame-shaped array is what lets
        the bounding box become a first-class concept without every existing caller changing: a
        comp with no overscan produces a byte-identical result to the pre-bounding-box evaluator.
        """
        return self.evaluate_raster(doc, target, cancel, frame, tier).to_display()

    def evaluate_raster(self, doc, target=None, cancel: threading.Event | None = None,
                        frame=None, tier=1, typed=False, return_digest=False,
                        cache_fractional=False):
        """Evaluate `target` at one timeline frame, optionally at a proxy tier.

        `frame` is an argument rather than ambient state on purpose: a clip-based timeline maps one
        timeline frame onto a different source frame per clip, so nothing may reach for a global
        playhead. See docs/TIME_MODEL.md. Omitting it uses the document's stored current frame.

        `cache_fractional`, when true, lets THIS call's result enter and be served from the memory
        cache even though `frame` is fractional. Every other caller of a fractional frame (Kronos,
        OFlow, MotionBlur/2D/3D, VectorGenerator, TimeWarp's own subframe blend) keeps the
        established "ephemeral" contract -- a scrub position is rarely revisited, so caching every
        one it passes through would spend budget for little reuse. TimeBlur's shutter subframes are
        different: the tile executor asks for the *same* node at the *same* outer frame, tile after
        tile, so the very same subframe positions repeat within one compose and across a second one
        at that frame. Only TimeBlur's own nested call (`kind == "TimeBlur"` below) passes this
        true; see docs/TIME_MODEL.md.

        `tier` is an argument for the same reason, and additionally because export must be able to
        ask for tier 1 while the viewer is showing tier 4 (contract clause C3). It is deliberately
        not stored in the document: a proxy setting is how one artist is looking at a comp right
        now, not a property of the comp.

        `return_digest`, when true, returns `(raster, digest)` instead of `raster`. This is how
        TimeOffset/FrameHold/Retime fold a *nested* evaluation's content digest into their own
        cache key: each is a clip-shaped producer whose input is evaluated at a remapped frame (a
        recursive `evaluate_raster` call, exactly the pattern TIME_MODEL.md's "clip timeline"
        section sketches), and its digest must reflect what that nested call actually resolved to
        at that frame — not the outer walk's `hashes[source]`, which was computed at the wrong
        (document) frame and would miss an edit to a keyframe elsewhere in the curve.
        """
        tier = int(tier)
        if tier not in tiers.PROXY_TIERS:
            raise ValueError(f"Unsupported proxy tier {tier}; expected one of {tiers.PROXY_TIERS}")
        target = target or doc["view"]
        if target is None:
            raise ValueError("Select a node and press 1 to view it")
        if frame is None:
            frame = doc.get("time", {}).get("current", 1)
        frame = float(frame) if isinstance(frame, float) and not frame.is_integer() else int(frame)
        fractional_frame = isinstance(frame, float)
        # Groups are expanded into the equivalent plain graph here, so the walk below (and every
        # cache key it makes) never sees one. See groups.py.
        doc, target = groups.flatten_groups(doc, target)
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
                from .core import bypass_slot
                slot = bypass_slot(nodes[key])
                inputs = [] if slot is None else [nodes[key]["inputs"][slot]]
            elif nodes[key]["type"] in _TIME_REMAP_KINDS + ("TimeBlur", "TimeEcho", "TimeWarp"):
                # This node's input is fetched by a nested `evaluate_raster` call at a remapped
                # frame (below), not from `values[source]` computed by this walk at `frame` --
                # walking into it here would only evaluate and cache it at the wrong frame,
                # uselessly (a cache miss neither this node nor anything else consumes) and, for
                # an animated source, would defeat exactly the reuse FrameHold exists to give a
                # scrub: `hashes[source]` would churn with the outer frame even while
                # `effective_frame` — and so the real result — stays put.
                inputs = []
            elif nodes[key]["type"] == "Profile":
                # Profile (below) times its own nested `evaluate_raster` call on its "image" input
                # so it can measure that call's wall time and `self.hits` delta. Walking into the
                # input here too would pre-compute and cache it before Profile's own call ever
                # ran, making every measurement report a 0 ms, all-cache-hit subgraph regardless
                # of what actually happened -- exactly the same reason TimeBlur is excluded above.
                inputs = []
            stack.extend((source, False) for source in inputs if source is not None)
        values, hashes = {}, {}
        from . import fluid3d, flip3d, particles

        def reads_lazily(key):
            """True when every reader of particle node `key` is an enabled particle cache/pass
            node, so `key` need not solve: the cache (or the force, which carries the run on and solves
            the whole chain itself when it is not read lazily too) does the solving."""
            readers = [other for other in order if key in nodes[other]["inputs"].values()]
            return bool(readers) and all(
                (nodes[other]["type"] in ("ParticleCache3D", "FluidWhitewater3D") and not nodes[other]["disabled"])
                or nodes[other]["type"] in particles.FORCE_KINDS for other in readers)
        for key in order:
            if cancel and cancel.is_set():
                raise Cancelled()
            node = nodes[key]
            kind = node["type"]
            if kind == "Input":
                raise ValueError(f"{node['name']}: an Input node only works inside a Group")
            if kind == "Profile" and not node["disabled"]:
                image_input = node["inputs"]["image"]
                hits_before = self.hits
                start = time.perf_counter()
                values[key] = self.evaluate_raster(doc, image_input, cancel=cancel, frame=frame,
                                                   tier=tier, cache_fractional=cache_fractional)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                self._record_profile(key, frame, elapsed_ms, self.hits - hits_before)
                continue
            active_inputs = node["inputs"]
            if node["disabled"]:
                from .core import bypass_slot
                slot = bypass_slot(node)
                active_inputs = {} if slot is None else {slot: node["inputs"][slot]}
            sources = list(active_inputs.values())
            # Resolve animation curves onto a per-frame params copy. node["params"] is the stored
            # base value; animation is an overlay that never mutates it. Static nodes (no curves)
            # resolve to a shallow copy that compares equal under json.dumps, so existing caches
            # keep their keys. See docs/ANIMATION.md.
            from .core import (SPECS as _SPECS, LIMITS as _LIMITS, OUTPUT_TYPES, GEOMETRY_TYPES,
                               _XFORM as _IDENTITY_XFORM, DRAW_KINDS as _DRAW_KINDS)
            from .animation import resolve_params as _resolve_params
            node_curves = doc.get("animation", {}).get("curves", {}).get(key)
            params = _resolve_params(node, node_curves, frame, _SPECS[kind]["params"], _LIMITS)
            if kind in ("OCIOColorspace", "OCIODisplay", "OCIOFileTransform", "OCIOLookTransform", "OCIOLogConvert", "Colorspace") and not params.get("config"):
                params["config"] = doc.get("settings", {}).get("color", {}).get("config", "")
            # Pixel-unit parameters are scaled in the same pass that shrinks the sources, so a blur
            # radius or a crop rectangle means the same thing at every tier (clause C3). Generated
            # sources therefore produce small pixels rather than being rendered full size and
            # shrunk, which is the difference between a tier that saves work and one that does not.
            # Scaling runs *after* curve resolution: a pixel-unit parameter must be scaled from the
            # value this frame actually uses, or an animated blur radius would proxy at its base.
            params = tiers.scale_params(kind, params, tier)
            if kind == "ContactSheet":
                params["_contact_labels"] = [nodes[node["inputs"].get(f"clip{i}")]["name"]
                                             if node["inputs"].get(f"clip{i}") in nodes else f"clip{i}"
                                             for i in range(16)]
            # Structured payloads (schema v8) follow exactly the same two steps as parameters, in
            # the same order and for the same reasons: scale the pixel units to this tier, then
            # resolve every animatable scalar at this frame. `scale_node_data` scales curve key
            # values too, so scaling first leaves a curve a curve and the order is not observable.
            payload = tiers.scale_node_data(kind, doc.get("node_data", {}).get(key), tier)
            data = None
            if kind == "Flare" and params.get("tracker_id"):
                linked_id = params["tracker_id"]
                linked = nodes[linked_id]
                linked_tracks = doc.get("node_data", {}).get(linked_id, {}).get("tracks", [])
                track = linked_tracks[params["track_index"]]
                reference_frame = int(linked["params"].get("reference_frame", 1))
                delta_x = shapes.resolve_scalar(track["x"], frame, "x") - shapes.resolve_scalar(track["x"], reference_frame, "x")
                delta_y = shapes.resolve_scalar(track["y"], frame, "y") - shapes.resolve_scalar(track["y"], reference_frame, "y")
                params["position_x"] += delta_x / tier
                params["position_y"] += delta_y / tier
            if kind == "Roto":
                data = shapes.resolve_shapes(payload, frame)
            elif kind in ("SplineWarp", "GridWarp", "GridWarpTracker"):
                if kind == "GridWarpTracker":
                    tracker_id = params.get("tracker_id", "")
                    tracker_node = nodes.get(tracker_id)
                    tracks = doc.get("node_data", {}).get(tracker_id, {}).get("tracks", [])
                    indices = [int(i.strip()) for i in str(params.get("track_indices", "0,1,2,3")).split(",") if i.strip()]
                    selected = [tracks[i] for i in indices if 0 <= i < len(tracks)]
                    ref = int(params.get("reference_frame", 1))
                    if params.get("drive", "tracker") == "tracker" and tracker_node is not None and selected:
                        ref_pts = np.array([[shapes.resolve_scalar(t["x"], ref, "x"), shapes.resolve_scalar(t["y"], ref, "y")] for t in selected], dtype=np.float64)
                        first = int(doc.get("time", {}).get("first", int(frame)))
                        history = []
                        latest = [None] * len(selected)
                        for candidate in range(int(frame), first - 1, -1):
                            positions = [None] * len(selected)
                            for j, track in enumerate(selected):
                                if shapes.resolve_scalar(track.get("enabled", 1), candidate, "enabled") >= .5:
                                    positions[j] = [shapes.resolve_scalar(track["x"], candidate, "x"),
                                                    shapes.resolve_scalar(track["y"], candidate, "y")]
                                    if latest[j] is None:
                                        latest[j] = positions[j]
                            history.append((candidate, positions))
                        active = [j for j, point in enumerate(latest) if point is not None]
                        if active:
                            data = {"tracker_points": (ref_pts[active].tolist(),
                                                       np.asarray([latest[j] for j in active]).tolist()),
                                    "tracker_indices": active,
                                    "tracker_history": [(f, [pts[j] for j in active]) for f, pts in history]}
                else:
                    data = shapes.resolve_warp_data(kind, payload, frame)
            elif kind == "RotoPaint":
                data = shapes.resolve_paint_items(payload, frame)
                raw_items = (payload or {}).get("items", [])
                for item, stored in zip(data, raw_items):
                    follow = stored.get("follow_track") if stored.get("kind") == "stroke" else None
                    if follow is None:
                        continue
                    tracker_payload = doc.get("node_data", {}).get(follow["node_id"], {"tracks": []})
                    track = tracker_payload["tracks"][follow["track_index"]]
                    current = shapes.resolve_scalar(track["x"], frame, "x"), shapes.resolve_scalar(track["y"], frame, "y")
                    life = stored["lifetime"]
                    reference_frame = int(life.get("first", doc["time"]["first"]))
                    enabled_now = shapes.resolve_scalar(track["enabled"], frame, "enabled") >= 0.5
                    enabled_ref = shapes.resolve_scalar(track["enabled"], reference_frame, "enabled") >= 0.5
                    reference = (shapes.resolve_scalar(track["x"], reference_frame, "x"),
                                 shapes.resolve_scalar(track["y"], reference_frame, "y"))
                    item["_track_delta"] = ([(current[0] - reference[0]) / tier,
                                              (current[1] - reference[1]) / tier]
                                             if enabled_now and enabled_ref else [0.0, 0.0])
            elif kind in ("Tracker", "Stabilize"):
                # Tracked geometry is merged into params rather than carried beside
                # them, so every later stage — window, kernel, region rule, digest — sees an
                # ordinary Transform and cannot treat the two differently by accident.
                params = {**params, **tracker.solve(payload, frame, params,
                                                      force_stabilise=(kind == "Stabilize"))}
            # Only required slots (those listed in SPECS[kind]["inputs"]) must be wired; optional
            # slots — like the new "mask" input on image-filter nodes — are allowed to be None and
            # the kernel treats that as identity (mask.a = 1, no extra gating).
            required = set(_SPECS.get(kind, {}).get("inputs", []))
            if kind == "Generate" and active_inputs.get("image") is None:
                raise ValueError("Generate: connect a source plate image")
            for slot, source in active_inputs.items():
                if source is None and slot in required:
                    raise ValueError(f"{node['name']}: connect required input(s)")
            # 3D scene values are typed runtime objects rather than image rasters. They stay on
            # the same graph/evaluation boundary, but are deliberately reference-rendered here:
            # the only value that crosses back into the existing 2D graph is Render3D's Raster.
            if OUTPUT_TYPES.get(kind, "image") != "image" or kind == "Render3D":
                from . import scene3d
                fingerprint = None
                if kind == "ReadSplat3D" and not node["disabled"]:
                    from . import splats
                    if not params["splat_path"]:
                        raise ValueError("ReadSplat3D: choose a splat file")
                    fingerprint = splats.fingerprint(params["splat_path"])
                if kind == "ReadGeo3D" and not node["disabled"]:
                    fingerprint = scene3d.obj_fingerprint(params["geo_path"])
                if kind in ("ReadUSD3D", "ReadUSDCamera3D") and not node["disabled"]:
                    from . import usdio
                    try:
                        fingerprint = [usdio.fingerprint(params["usd_path"]), frame]
                    except RuntimeError as error:
                        raise ValueError(str(error)) from None
                if kind == "ReadGLTF3D" and not node["disabled"]:
                    from . import gltfio
                    fingerprint = gltfio.fingerprint(params["gltf_path"])
                if kind in ("ReadAlembic3D", "ReadAlembicCamera3D") and not node["disabled"]:
                    from . import alembicio
                    seconds = frame / doc["time"]["fps"]
                    fingerprint = [alembicio.fingerprint(params["abc_path"]), frame, seconds]
                if kind == "ReadVDB3D" and not node["disabled"]:
                    from . import vdbio
                    # The resolved file of this frame is the fingerprint, so a sequence re-keys per frame.
                    fingerprint = vdbio.fingerprint(vdbio.frame_path(
                        params["vdb_path"], frame + int(params["frame_offset"])))
                stream = None
                if kind == "ParticleEmitter3D" and not node["disabled"]:
                    stream = particles.build_stream(self, doc, key, node, cancel)
                    fingerprint = [stream.run, frame]
                elif kind == "ParticleCache3D" and not node["disabled"]:
                    stream = getattr(values[node["inputs"]["particles"]], "stream", None)
                    fingerprint = [None if stream is None else stream.run, frame]
                elif kind in particles.FORCE_KINDS:
                    stream = getattr(values[node["inputs"]["particles"]], "stream", None)
                    if isinstance(stream, flip3d.LiquidStream):
                        stream = None       # a liquid takes FluidForce3D nodes; particle forces pass it on unchanged
                    if stream is not None and not node["disabled"]:
                        stream = particles.extend_stream(stream, doc, key, node, self, cancel)
                    fingerprint = [None if stream is None else stream.run, frame]
                fluid = None
                if kind in fluid3d.CHAIN_KINDS:
                    slot = node["inputs"].get("fluid")
                    fluid = fluid3d.chain_for(self, doc, key, node, None if slot is None else values[slot], cancel)
                    fingerprint = [fluid.run]
                elif kind == "FluidSolver3D" and not node["disabled"]:
                    fluid = fluid3d.build_stream(doc, key, node, values[node["inputs"]["fluid"]])
                    fingerprint = [fluid.run, frame]
                elif kind == "FluidCache3D" and not node["disabled"]:
                    fluid = getattr(values[node["inputs"]["volume"]], "stream", None)
                    fingerprint = [None if fluid is None else fluid.run, frame]
                elif kind == "FluidUpres3D" and not node["disabled"]:
                    fluid = getattr(values[node["inputs"]["volume"]], "stream", None)
                    fingerprint = [None if fluid is None else fluid.run, frame]
                elif kind == "FluidLiquidSolver3D" and not node["disabled"]:
                    fluid = flip3d.build_stream(doc, key, node, values[node["inputs"]["fluid"]])
                    fingerprint = [fluid.run, frame, self._liquid_feedback_versions.get(fluid.run, 0)]
                elif kind == "FluidWhitewater3D" and not node["disabled"]:
                    fluid = getattr(values[node["inputs"]["particles"]], "stream", None)
                    fingerprint = [None if fluid is None else fluid.run, frame, params]
                elif kind == "RigidSolver3D":
                    fingerprint = [frame]
                motion_moments = motion_later = None
                if kind == "Render3D" and not node["disabled"]:
                    motion_moments, motion_later, fingerprint = self._motion_inputs(
                        doc, node, params, frame, tier, cancel,
                        (values[node["inputs"]["scene"]], values[node["inputs"]["camera"]]))
                digest = hashlib.sha256(json.dumps([kind, params, node["disabled"],
                                                     [hashes[s] if s is not None else None for s in sources],
                                                     fingerprint, tier, data], sort_keys=True).encode()).hexdigest()
                hashes[key] = digest
                if kind in GEOMETRY_TYPES:
                    # A disabled geometry node contributes nothing rather than passing its texture on.
                    texture = values[sources[0]] if sources and sources[0] is not None else None
                    value = None if node["disabled"] else scene3d.geometry_from_node(
                        {"type": kind, "params": params, "name": node["name"]},
                        None if texture is None else texture.to_display())
                elif kind == "RigidBody3D":
                    if node["disabled"]:
                        value = None
                    else:
                        from .rigid3d import RigidBody3D, convex_hull_triangles
                        attached = []
                        for slot in ("geometry", *(f"part{i}" for i in range(8))):
                            source_key = node["inputs"].get(slot)
                            if source_key is None:
                                continue
                            source_value = values[source_key]
                            if isinstance(source_value, scene3d.Geometry):
                                attached.append(source_value)
                            elif isinstance(source_value, scene3d.Scene):
                                attached.extend(scene3d.resolve_instances(source_value).geometries)
                        size = np.array([params["size_x"], params["size_y"], params["size_z"]], np.float64)
                        if attached:
                            points = np.concatenate([g.world_matrix()[:3, :3] @ g.vertices.T +
                                                     g.world_matrix()[:3, 3:4] for g in attached], axis=1)
                            extent = points.max(axis=1) - points.min(axis=1)
                            size = np.maximum(extent, 1e-4)
                        value = RigidBody3D(
                            shape=params["rigid_shape"], position=(params["tx"], params["ty"], params["tz"]),
                            rotation=(params["rx"], params["ry"], params["rz"]),
                            size=size, velocity=(params["velocity_x"], params["velocity_y"], params["velocity_z"]),
                            angular_velocity=(params["angular_velocity_x"], params["angular_velocity_y"],
                                              params["angular_velocity_z"]),
                            torque=(params["torque_x"], params["torque_y"], params["torque_z"]),
                            density=params["density"], mass=params["mass"], friction=params["friction"],
                            restitution=params["restitution"], dynamic=bool(params["dynamic"]), geometry=tuple(attached))
                        if params["rigid_shape"] in ("convex", "compound") and not attached:
                            raise ValueError(f"RigidBody3D {params['rigid_shape']} shape requires connected geometry")
                        if attached:
                            inverse_rotation = value.rotation_matrix().T
                            parts = []
                            for g in attached:
                                matrix = g.world_matrix()
                                world_vertices = g.vertices.astype(np.float64) @ matrix[:3, :3].T + matrix[:3, 3]
                                local = (inverse_rotation @ (world_vertices - value.position).T).T
                                if params["rigid_shape"] in ("convex", "compound"):
                                    local, triangles = convex_hull_triangles(local, g.triangles)
                                else:
                                    triangles = g.triangles
                                parts.append((local, triangles))
                            value.collision_parts = tuple(parts)
                elif kind == "RigidSolver3D":
                    if node["disabled"]:
                        value = scene3d.Scene()
                    else:
                        from .rigid3d import RigidSolver3D as Solver, liquid_reaction
                        bodies = [copy.deepcopy(values[source]) for slot, source in node["inputs"].items()
                                  if slot.startswith("body") and source is not None and values[source] is not None]
                        liquid_source = node["inputs"].get("liquid")
                        liquid = None if liquid_source is None else values[liquid_source]
                        waterline = None
                        if liquid is not None and len(getattr(liquid, "positions", ())):
                            positions = np.asarray(liquid.positions, dtype=np.float64)
                            matrix = np.asarray(getattr(liquid, "matrix", np.eye(4)), dtype=np.float64)
                            waterline = float(np.max((matrix[:3, :3] @ positions.T).T[:, 1] + matrix[1, 3]))
                        solver_id = hashlib.sha256(json.dumps(
                            [key, params, [hashes[source] for slot, source in node["inputs"].items()
                                           if source is not None and slot.startswith("body")]],
                            sort_keys=True).encode()).hexdigest()
                        sim = self._rigid_solvers.get(solver_id)
                        if sim is None:
                            sim = Solver(bodies, gravity=(params["gravity_x"], params["gravity_y"], params["gravity_z"]),
                                         fps=doc["time"]["fps"], substeps=params["substeps"],
                                         floor_y=params["floor_y"] if params["floor"] == "on" else None,
                                         iterations=params["iterations"], sleep_threshold=params["sleep_threshold"],
                                         sleep_frames=params["sleep_time"], liquid_surface_y=waterline,
                                         liquid_density=params["liquid_density"])
                            if len(self._rigid_solvers) >= 64:
                                self._rigid_solvers.pop(next(iter(self._rigid_solvers)))
                            self._rigid_solvers[solver_id] = sim
                        sim.solve_frame(max(0, int(math.floor(float(frame))) - 1), cancel)
                        geometries = []
                        for body in sim.bodies:
                            meshes = list(body.geometry)
                            generated = not meshes
                            if not meshes:
                                primitive = "Sphere3D" if body.shape == "sphere" else "Cube3D"
                                from .core import SPECS
                                p = dict(SPECS[primitive]["params"])
                                p.update({"cube_size": 2.0, "sphere_radius": 1.0, "rows": 16, "columns": 32,
                                          "sx": float(body.size[0] / 2), "sy": float(body.size[1] / 2),
                                          "sz": float(body.size[2] / 2)})
                                meshes = [scene3d.geometry_from_node({"type": primitive, "params": p})]
                            current_rotation = body.rotation_matrix()
                            if generated:
                                body_matrix = np.eye(4)
                                body_matrix[:3, :3] = current_rotation
                                body_matrix[:3, 3] = body.position
                            else:
                                initial_rotation = scene3d.Transform3D(
                                    rotation=scene3d.Vec3(*map(float, body.initial_rotation))).matrix()[:3, :3]
                                delta_rotation = current_rotation @ initial_rotation.T
                                body_matrix = np.eye(4)
                                body_matrix[:3, :3] = delta_rotation
                                body_matrix[:3, 3] = body.position - delta_rotation @ body.initial_position
                            geometries.extend(replace(g, parent=(body_matrix @ g.parent).astype(np.float32))
                                              for g in meshes)
                        feedback_state = None
                        if liquid is not None and isinstance(getattr(liquid, "stream", None), flip3d.LiquidStream):
                            feedback_store = self._sim_memory
                            for candidate in reversed(tuple(self._sim_stores.values())):
                                if candidate.get(liquid.stream.run, liquid.frame) is not None:
                                    feedback_store = candidate
                                    break
                            feedback_before = feedback_store.get(liquid.stream.run, liquid.frame)
                            feedback_state = flip3d.apply_rigid_feedback(
                                liquid.stream, liquid.frame, feedback_store, sim.bodies, solver_id,
                                gravity=params["gravity_y"])
                            if feedback_before is not None and feedback_state != feedback_before:
                                run = liquid.stream.run
                                self._liquid_feedback_versions[run] = self._liquid_feedback_versions.get(run, 0) + 1
                        if liquid is None:
                            coupled_liquid = None
                        elif feedback_state is not None:
                            coupled_liquid = replace(liquid, velocities=feedback_state.arrays["velocity"])
                        else:
                            coupled_liquid = liquid_reaction(liquid, sim.bodies, gravity=params["gravity_y"])
                        value = scene3d.Scene(geometries=tuple(geometries),
                                              particles=() if coupled_liquid is None else (coupled_liquid,))
                elif kind == "TransformGeo3D":
                    # Disabled bakes nothing: the geometry passes through exactly as bypass_slot
                    # says (its one required input), matching a disabled 2D Transform.
                    source = values[node["inputs"]["geo"]]
                    value = source if node["disabled"] else scene3d.transform_geometry(
                        source, scene3d._transform_from(params))
                elif kind == "MergeGeo3D":
                    # Disabled passes the first wired geometry through untouched (bypass_slot), and
                    # an empty merge is an empty geometry, never a crash.
                    if node["disabled"]:
                        value = next((v for v in (values[s] for s in sources if s is not None)
                                      if v is not None), scene3d.empty_geometry())
                    else:
                        wired = [values[s] for s in sources if s is not None]
                        value = scene3d.merge_geometry(wired, scene3d._transform_from(params))
                elif kind == "ParticleEmitter3D":
                    # Disabled passes the emission geometry (bypass_slot "geo"), or nothing when unwired.
                    if node["disabled"]:
                        source = node["inputs"].get("geo")
                        value = values[source] if source is not None else None
                    elif key != target and reads_lazily(key):
                        # Only ParticleCache3D and force nodes read this emitter: the cache solves the run
                        # through its own persistent store, so the emitter hands over the run without
                        # solving it twice.
                        value = particles.placeholder_instance(stream, frame)
                    else:
                        state = particles.solve_frame(stream, frame, self._sim_memory, cancel)
                        value = particles.instance_from_state(state, stream, frame)
                elif kind in particles.FORCE_KINDS:
                    incoming = values[node["inputs"]["particles"]]
                    if stream is None:
                        value = incoming
                    elif key != target and reads_lazily(key):
                        value = replace(incoming, stream=stream, frame=int(frame))
                    else:
                        state = particles.solve_frame(stream, frame, self._sim_memory, cancel)
                        value = replace(particles.instance_from_state(state, stream, frame),
                                        matrix=incoming.matrix, render_as=incoming.render_as,
                                        size_scale=incoming.size_scale, texture=incoming.texture,
                                        foam_density=incoming.foam_density, spray_size=incoming.spray_size)
                elif kind == "ParticleRender3D":
                    incoming = values[node["inputs"]["particles"]]
                    if node["disabled"]:
                        value = incoming
                    else:
                        image = node["inputs"].get("image")
                        value = replace(incoming, render_as=params["representation"],
                                        size_scale=params["size_scale"],
                                        foam_density=float(params.get("foam_density", 1.0)),
                                        spray_size=float(params.get("spray_size", 1.0)),
                                        texture=None if image is None else values[image].to_display())
                        # R7 of 7: material and attribute ramps, baked once here (never inside the solve).
                        value = scene3d.apply_particle_look(value, params)
                elif kind == "Instance3D":
                    # Disabled passes the points through untouched (bypass_slot "points"): whatever
                    # they were (particles or geometry), not a Scene of instances.
                    if node["disabled"]:
                        source = node["inputs"].get("points")
                        value = values[source] if source is not None else None
                    else:
                        points_value = values[node["inputs"]["points"]]
                        instance_slot = node["inputs"].get("instance")
                        instance_value = None if instance_slot is None else values[instance_slot]
                        instance_set = replace(
                            scene3d.instances_from_node(points_value, instance_value, params), node_key=key)
                        value = scene3d.Scene(instances=(instance_set,))
                elif kind == "ParticleCache3D":
                    incoming = values[node["inputs"]["particles"]]
                    if node["disabled"] or stream is None:
                        value = incoming
                    else:
                        store = self.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
                        state = particles.solve_frame(stream, frame, store, cancel)
                        value = replace(particles.instance_from_state(state, stream, frame),
                                        matrix=incoming.matrix, render_as=incoming.render_as,
                                        size_scale=incoming.size_scale, texture=incoming.texture,
                                        foam_density=incoming.foam_density, spray_size=incoming.spray_size)
                elif kind in fluid3d.CHAIN_KINDS:
                    value = fluid
                elif kind == "FluidSolver3D":
                    if node["disabled"]:
                        value = None
                    elif key != target and all(nodes[other]["type"] == "FluidCache3D" and not nodes[other]["disabled"]
                                               for other in order if key in nodes[other]["inputs"].values()):
                        # only enabled FluidCache3D nodes read this solver: the cache solves through its own store
                        value = fluid3d.placeholder_volume(fluid, frame)
                    else:
                        state = fluid3d.solve_frame(fluid, frame, self._sim_memory, cancel)
                        value = fluid3d.volume_from_state(state, fluid, frame)
                elif kind == "FluidLiquidSolver3D":
                    if node["disabled"]:
                        value = flip3d.empty_instance()
                    elif key != target and all(nodes[other]["type"] in ("ParticleCache3D", "FluidWhitewater3D") and not nodes[other]["disabled"]
                                               for other in order if key in nodes[other]["inputs"].values()):
                        # only enabled ParticleCache3D nodes read this liquid: the cache solves through its own store
                        value = particles.placeholder_instance(fluid, frame)
                    else:
                        state = flip3d.solve_frame(fluid, frame, self._sim_memory, cancel)
                        value = flip3d.instance_from_state(state, fluid, frame)
                elif kind == "FluidSurface3D":
                    incoming = values[node["inputs"]["particles"]]
                    if node["disabled"] or incoming is None:
                        value = scene3d.empty_geometry()
                    else:
                        radius = max(0, int(params.get("temporal_smoothing", 0)))
                        temporal = []
                        if radius and isinstance(getattr(incoming, "stream", None), flip3d.LiquidStream):
                            for offset in range(-radius, radius + 1):
                                if not offset:
                                    continue
                                sample_frame = incoming.frame + offset
                                state = flip3d.solve_frame(incoming.stream, sample_frame, self._sim_memory, cancel)
                                temporal.append(flip3d.instance_from_state(state, incoming.stream, sample_frame))
                        value = replace(flip3d.surface_geometry(incoming, params, temporal),
                                        **scene3d.material_fields(params))
                elif kind == "FluidFoam3D":
                    incoming = values[node["inputs"]["particles"]]
                    value = flip3d.empty_instance() if node["disabled"] or incoming is None else \
                        flip3d.foam_instance(incoming, params)
                elif kind == "FluidWhitewater3D":
                    incoming = values[node["inputs"]["particles"]]
                    if node["disabled"]:
                        value = incoming
                    elif incoming is None or fluid is None:
                        value = flip3d.empty_instance()
                    else:
                        from . import simcache, whitewater
                        white_solver = whitewater.FluidWhitewater3D(params, params["seed"], fluid.fps,
                                                                    getattr(fluid.chain, "colliders", ()))
                        run = simcache.run_key(fluid.run, {"kind": "FluidWhitewater3D", "params": params,
                                                           "format": 1})
                        store = self.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
                        def initial_whitewater(seed):
                            return _whitewater_cache_state(white_solver.initial_state(seed))
                        def whitewater_step(previous, solve_frame, substep, seed):
                            liquid_state = flip3d.solve_frame(fluid, solve_frame, store, cancel)
                            liquid_frame = flip3d.instance_from_state(liquid_state, fluid, solve_frame)
                            previous_state = _whitewater_state(previous)
                            result = white_solver.step(previous_state, liquid_frame, solve_frame, substep, seed)
                            return _whitewater_cache_state(result)
                        solved = simcache.solve_to_frame(store, run, int(frame), fluid.start_frame, 1,
                                                         int(params["seed"]), initial_whitewater, whitewater_step, cancel)
                        value = whitewater.instance_from_state(_whitewater_state(solved), incoming, frame)
                elif kind == "FluidCache3D":
                    incoming = values[node["inputs"]["volume"]]
                    if node["disabled"] or fluid is None:
                        value = incoming
                    else:
                        store = self.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
                        value = fluid3d.cached_volume(fluid, frame, store, cancel, params["cache_precision"],
                                                      params["cache_channels"])
                elif kind == "FluidUpres3D":
                    incoming = values[node["inputs"]["volume"]]
                    if node["disabled"] or incoming is None:
                        value = incoming
                    else:
                        from . import fluid_upres
                        store = self.sim_store(params["cache_memory_mb"], params["cache_disk_mb"])
                        value = None
                        first = min(int(getattr(fluid, "start_frame", frame)), int(frame)) \
                            if fluid is not None else int(frame)
                        for up_frame in range(first, int(frame) + 1):
                            if cancel is not None and hasattr(cancel, "check"):
                                cancel.check()
                            coarse = incoming if up_frame == int(frame) else fluid3d.cached_volume(
                                fluid, up_frame, self._sim_memory, cancel, "float32", "all")
                            guide_velocity = coarse.velocity
                            if fluid is not None and guide_velocity is not None:
                                next_coarse = fluid3d.cached_volume(
                                    fluid, up_frame + 1, self._sim_memory, cancel, "float32", "all")
                                if next_coarse.velocity is not None:
                                    guide_velocity = 0.5 * (guide_velocity + next_coarse.velocity)
                            value = fluid_upres.cached_upres(coarse, params, up_frame, store, cancel,
                                                             guide_velocity, value)
                elif kind == "Normals3D":
                    source = values[node["inputs"]["geo"]]
                    value = source if node["disabled"] or source is None else scene3d.normals_from_node(source, params)
                elif kind == "DisplaceGeo3D":
                    source = values[node["inputs"]["geo"]]
                    image = None
                    if not node["disabled"] and node["inputs"].get("image") is not None:
                        image = values[node["inputs"]["image"]]
                    value = source if node["disabled"] or source is None else scene3d.displace_geometry(
                        source, None if image is None else image.to_display(),
                        params["displace_scale"], params["displace_offset"],
                        params["displace_channel"], bool(params["recompute_normals"]))
                elif kind == "Shrinkwrap3D":
                    source = values[node["inputs"]["target"]]
                    if node["disabled"] or source is None:
                        value = source
                    else:
                        proxy_slot = node["inputs"].get("proxy")
                        proxy = None if proxy_slot is None else values[proxy_slot]
                        value = scene3d.shrinkwrap_geometry(source, proxy, params)
                elif kind == "ReadSplat3D":
                    value = scene3d.Scene() if node["disabled"] else scene3d.Scene(splats=(
                        scene3d.SplatInstance(self._delit_cloud(splats.load_cloud_cached(
                            params["splat_path"], params["splat_orientation"], params["splat_colorspace"]),
                            params, cancel),
                            scene3d._transform_from(params).matrix(), params["splat_sh_degree"],
                            params["splat_opacity"], params["splat_scale"], params.get("splat_relight", 0.0),
                            params.get("splat_shadow_catch", 0.0),
                            params.get("splat_cast_shadows", "on") == "on",
                            params.get("splat_specular", 0.0),
                            int(params.get("splat_normal_smoothing", 0)),
                            params.get("splat_use_intrinsics", "on") == "on",
                            float(params.get("splat_metallic", 0.0)), float(params.get("splat_roughness", 1.0)),
                            float(params.get("splat_intrinsics_mix", 1.0)),
                            int(params.get("splat_reflection_samples", 0)),
                            int(params.get("splat_indirect_samples", 0)),
                            float(params.get("splat_indirect_distance", 1.0)),
                            float(params.get("splat_denoise", 0.0)),
                            str(params.get("splat_quality", "medium")), name=node["name"],
                            light_link=scene3d.light_link_from_params(params)),))
                elif kind in ("ReadUSD3D", "ReadUSDCamera3D"):
                    from . import usdio
                    try:
                        if kind == "ReadUSD3D":
                            value = (scene3d.Scene() if node["disabled"] else
                                     usdio.load_scene(params["usd_path"], frame, params["usd_root"]))
                        else:
                            value = (scene3d.Camera() if node["disabled"] else
                                     usdio.load_camera(params["usd_path"], frame, params["usd_camera"]))
                    except RuntimeError as error:
                        raise ValueError(str(error)) from None
                elif kind == "ReadGLTF3D":
                    from . import gltfio
                    value = (scene3d.Scene() if node["disabled"] else
                             gltfio.load_scene(params["gltf_path"], params["gltf_root"]))
                elif kind in ("ReadAlembic3D", "ReadAlembicCamera3D"):
                    from . import alembicio
                    if kind == "ReadAlembic3D":
                        value = (scene3d.Scene() if node["disabled"] else
                                 alembicio.load_scene(params["abc_path"], seconds, params["abc_root"]))
                    else:
                        value = (scene3d.Camera() if node["disabled"] else
                                 alembicio.load_camera(params["abc_path"], seconds, params["abc_camera"]))
                elif kind == "ReadVDB3D":
                    # Disabled is an empty scene (a read has no input to pass through).
                    if node["disabled"]:
                        value = scene3d.Scene()
                    else:
                        from . import vdbio
                        volume = vdbio.load_volume(
                            vdbio.frame_path(params["vdb_path"], frame + int(params["frame_offset"])),
                            params["density_grid"], params["temperature_grid"], params["velocity_grid"],
                            params["voxel_scale"])
                        value = scene3d.Scene(volumes=(replace(
                            volume, matrix=scene3d._transform_from(params).matrix() @ volume.matrix),))
                elif kind == "Plume3D":
                    # Disabled contributes nothing (like Light3D): a volume has no input to pass through.
                    value = None if node["disabled"] else replace(
                        scene3d.analytic_plume(params["plume_resolution"], params["plume_seed"]),
                        matrix=scene3d._transform_from(params).matrix())
                elif kind == "Light3D":
                    image = None
                    if params["light_type"] == "Environment" and node["inputs"].get("image") is not None:
                        image = self._environment_map(values[node["inputs"]["image"]])
                    value = None if node["disabled"] else scene3d.light_from_node({"params": params, "name": node["name"]}, image)
                elif kind == "Camera3D":
                    value = scene3d.camera_from_node({"params": params})
                elif kind == "Project3D":
                    value = values[node["inputs"]["geometry"]]
                    if isinstance(value, scene3d.Geometry):
                        value = scene3d.Scene((value,))
                    elif value is None:  # disabled upstream geometry contributes nothing
                        value = scene3d.Scene()
                    if not node["disabled"]:
                        value = scene3d.apply_projection(value, scene3d.Projection(
                            camera=values[node["inputs"]["camera"]],
                            texture=values[node["inputs"]["image"]].to_display(),
                            outside=params["project_outside"], backfaces=params["project_backfaces"],
                            occlusion=params.get("project_occlusion", "off")))
                elif kind in ("WriteGeo3D", "WriteSplat3D", "WriteVDB3D"):
                    value = values[node["inputs"]["scene"]]
                elif kind == "Axis3D":
                    # Chaining is ordinary scene nesting: scene_from_node already flattens a Scene
                    # member and multiplies its own matrix onto each item's parent, in order, so two
                    # chained Axis3D compose exactly as Scene3D nesting does. Disabled uses the
                    # identity transform rather than going empty (bypass_slot passes "object"
                    # through), matching how a disabled 2D Transform passes its image unchanged.
                    source = node["inputs"]["object"]
                    member = values[source] if source is not None else None
                    members = [] if member is None else [member]
                    value = scene3d.scene_from_node(
                        {"params": _IDENTITY_XFORM if node["disabled"] else params, "name": node["name"]}, members)
                elif kind == "Scene3D":
                    slots = [node["inputs"].get(s) for s in _SPECS[kind]["optional_inputs"]]
                    members = [values[s] for s in slots if s is not None and values[s] is not None]
                    value = scene3d.Scene() if node["disabled"] else scene3d.scene_from_node(
                        {"params": params, "name": node["name"]}, members)
                elif digest in self.cache:
                    self.hits += 1
                    value = self.cache.pop(digest)
                    self.cache[digest] = value
                else:
                    self.misses += 1
                    if node["disabled"]:  # nothing sensible to pass through: a scene is not an image
                        values[key] = Raster.of(np.zeros((params["height"], params["width"], 4), np.float32))
                        continue
                    scene, camera = (values[node["inputs"][slot]] for slot in ("scene", "camera"))
                    if getattr(scene, "volumes", ()) and params.get("volumes", "on") == "off":
                        scene = replace(scene, volumes=())   # the knob makes every backend ignore them
                    if params.get("render_output", "rgba") == "motion":
                        from . import motionblur
                        later_scene, later_camera = motion_later
                        value = Raster.of(motionblur.motion_vectors(scene, camera, later_scene, later_camera,
                                                                    params["width"], params["height"]))
                        self._store(digest, value)
                        values[key] = value
                        continue
                    backend = params.get("render_backend", "cpu")
                    multichannel_backend = backend    # its volume passes run on the GPU; the other layers are CPU
                    if params.get("render_output", "rgba") in ("relight", "multichannel"):
                        chosen = set(scene3d.parse_passes(params.get("passes", scene3d.DEFAULT_PASSES))) \
                            if params["render_output"] == "multichannel" else set()
                        if backend == "gpu" and not (chosen & set(scene3d.VOLUME_OUTPUTS)) and params.get("render_mode") != "pathtrace":
                            what = "the relight bundle" if params["render_output"] == "relight" else "the multichannel"
                            raise ValueError(f"GPU Render3D unsupported: {what} output is CPU-only for now"
                                             + ("" if params["render_output"] == "relight"
                                                else " (only its volume passes run on the GPU)"))
                        backend = "cpu"
                    mode = params.get("render_mode", "raster")
                    args = (scene, camera, params["width"], params["height"],
                            (params["red"], params["green"], params["blue"], params["alpha"]))
                    if params.get("render_output", "rgba") == "multichannel":
                        from . import motionblur
                        motion_layer = None
                        if "motion" in scene3d.parse_passes(params.get("passes", scene3d.DEFAULT_PASSES)):
                            later_scene, later_camera = motion_later
                            motion_layer = motionblur.motion_vectors(scene, camera, later_scene, later_camera,
                                                                     params["width"], params["height"])
                        if motion_moments:
                            beauty, extra = motionblur.multichannel(
                                motion_moments, params["width"], params["height"], args[4],
                                passes=params.get("passes", scene3d.DEFAULT_PASSES), ambient=params["ambient"],
                                samples=params["samples"], cancel=cancel, mode=mode, progress=self.progress,
                                volume=_volume_settings(params), backend=multichannel_backend,
                                path=_path_settings(params, mode), motion_layer=motion_layer)
                        else:
                            beauty, extra = scene3d.render_multichannel(
                                *args, passes=params.get("passes", scene3d.DEFAULT_PASSES),
                                ambient=params["ambient"], samples=params["samples"], cancel=cancel, mode=mode,
                                progress=self.progress, volume=_volume_settings(params), backend=multichannel_backend,
                                path=_path_settings(params, mode), motion_layer=motion_layer)
                        value = Raster(beauty, layers={name: Raster.of(arr) for name, arr in extra.items()})
                        value = _add_cryptomatte(value, scene, camera, params, cancel, mode)
                        self._store(digest, value)
                        values[key] = value
                        continue
                    kwargs = dict(ambient=params["ambient"], samples=params["samples"],
                                  output=params.get("render_output", "rgba"), cancel=cancel, mode=mode)
                    background = (params["red"], params["green"], params["blue"], params["alpha"])

                    def draw(scene_at, camera_at):
                        """One raster or ray-traced image of `scene_at` seen by `camera_at` (one shutter time)."""
                        args_at = (scene_at, camera_at, params["width"], params["height"], background)
                        image = None
                        if backend != "cpu":
                            from . import gpu3d
                            if gpu3d.available():
                                try:
                                    image = gpu3d.render(*args_at, volume=_volume_settings(params), **kwargs)
                                except Cancelled:
                                    raise
                                except gpu3d.Unsupported as exc:
                                    if backend == "gpu":
                                        raise ValueError(f"GPU Render3D unsupported: {exc}") from exc
                                except Exception as exc:
                                    # Device loss, out of memory or an adapter error after available()
                                    # said yes: auto falls back to the CPU reference, gpu reports it.
                                    if backend == "gpu":
                                        raise ValueError(f"GPU Render3D failed: {exc}") from exc
                            elif backend == "gpu":
                                raise ValueError(f"GPU Render3D unavailable: {gpu3d.describe()}")
                        if image is None:
                            image = scene3d.render(*args_at, shadows=True, progress=self.progress,
                                                   volume=_volume_settings(params), **kwargs)
                        return image
                    output = params.get("render_output", "rgba")
                    raw_beauty = None
                    if params.get("denoise", "off") == "final" and mode != "pathtrace":
                        raise ValueError("Denoise final needs Render3D's path tracer mode (render_mode pathtrace)")
                    if mode == "pathtrace":
                        from . import pathtrace
                        # `denoise` "final" (step R1) runs the denoiser on the rgba output exactly as the "denoise"
                        # output does, with the same controls; `beauty_raw` keeps the unfiltered beauty as a layer.
                        final_denoise = params.get("denoise", "off") == "final" and output == "rgba"
                        if final_denoise:
                            output = "denoise"
                        denoise_kwargs = dict(
                            denoise_settings=pathtrace.denoise_settings_from_params(params),
                            history_key=key if params.get("denoise_temporal") else None
                        ) if output == "denoise" else {}
                        settings = pathtrace.settings_from_params(params)
                        keep_raw = final_denoise and bool(params.get("beauty_raw", 0))
                        if motion_moments:
                            rgba = pathtrace.render_motion(
                                motion_moments, params["width"], params["height"], background, params["ambient"],
                                output, settings, cancel=cancel,
                                progress=self.progress, backend=backend, volume=_volume_settings(params))
                            if keep_raw:    # the moments' own noisy beauty, same seeds as the filtered one's
                                raw_beauty = pathtrace.render_motion(
                                    motion_moments, params["width"], params["height"], background, params["ambient"],
                                    "rgba", settings, cancel=cancel, backend=backend, volume=_volume_settings(params))
                        else:
                            denoise_stats = {} if keep_raw else None
                            rgba = pathtrace.render(
                                scene, camera, params["width"], params["height"], background, params["ambient"],
                                output, settings,
                                cancel=cancel, progress=self.progress, backend=backend, volume=_volume_settings(params),
                                stats=denoise_stats, **denoise_kwargs)
                            if keep_raw:
                                raw_beauty = pathtrace.over_background(
                                    denoise_stats["beauty_raw"].astype(np.float64), background).astype(np.float32)
                    elif motion_moments and output not in scene3d.DATA_OUTPUTS:
                        from . import motionblur
                        drawn = [draw(scene_at, camera_at) for scene_at, camera_at in motion_moments]
                        rgba = motionblur.blend_bundle(drawn) if output == "relight" else motionblur.averaged(drawn)
                    else:
                        rgba = draw(scene, camera)
                    if params.get("render_output", "rgba") == "relight":
                        rgba, layers = rgba
                        value = Raster(rgba, layers={name: Raster.of(arr) for name, arr in layers.items()})
                    elif raw_beauty is not None:
                        value = Raster(rgba, layers={"beauty_raw": Raster.of(raw_beauty)})
                    else:
                        value = Raster.of(rgba)
                    value = _add_cryptomatte(value, scene, camera, params, cancel, mode)
                    self._store(digest, value)
                values[key] = value
                continue
            # Time enters the digest only where it changes the result. A Read resolves the concrete
            # file for this frame and fingerprints *that*; a still resolves to the same path at
            # every frame and keeps its cache entry, while a sequence naturally re-keys. Downstream
            # digests already fold in their inputs' hashes, so time-dependence propagates exactly as
            # far as it really reaches. See docs/TIME_MODEL.md.
            fingerprint = None
            clone_sources = {}
            if kind == "ReadBundle" and not node["disabled"]:
                from . import bundle
                fingerprint = bundle.fingerprint(params, frame)
            if kind == "ConditionedRead" and not node["disabled"]:
                fingerprint = []
                from .media import sequence_path
                generated_path = sequence_path(params["path"],
                                               int(frame) + int(params.get("frame_offset", 0)))
                for name, source_path in (("path", generated_path),
                                          ("manifest", params.get("manifest", "")),
                                          ("scene_state", params.get("scene_state", ""))):
                    try:
                        stat = Path(source_path).expanduser().stat()
                        fingerprint.extend((str(Path(source_path).expanduser().resolve()),
                                             stat.st_size, stat.st_mtime_ns))
                    except (OSError, KeyError):
                        fingerprint.extend((source_path, "missing"))
            if kind == "Read" and params["path"]:
                from .media import nearest_sequence_path, resolve_source_path
                source_frame = int(math.floor(frame + int(params.get("frame_offset", 0)) + 0.5))
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
            if kind == "Vectorfield" and params.get("cube_path"):
                try:
                    stat = Path(params["cube_path"]).expanduser().stat()
                    fingerprint = [str(Path(params["cube_path"]).expanduser().resolve()), stat.st_size, stat.st_mtime_ns]
                except OSError:
                    fingerprint = [params["cube_path"], "missing"]
            if kind == "OCIOFileTransform" and params.get("path"):
                try:
                    stat = Path(params["path"]).expanduser().stat()
                    fingerprint = [str(Path(params["path"]).expanduser().resolve()), stat.st_size, stat.st_mtime_ns]
                except OSError:
                    fingerprint = [params["path"], "missing"]
            if kind in _FRAME_METADATA_KINDS and not node["disabled"] and (
                    kind == "AddTimeCode" or any(metadata.uses_frame(str(v)) for v in params.values())):
                # These write the timeline frame into metadata or pixels, so the frame is part of
                # the result even though no upstream digest carries it.
                fingerprint = ["frame", frame]
            if kind == "Relight":
                # Sparse light slots are paired by index, so their positions affect the result.
                fingerprint = [slot for slot, source in active_inputs.items() if source is not None]
            remap_raster = None
            temporal_samples = None
            smartvector_layers = None
            smartvector_sequence = None
            smartvector_first = None
            smartvector_last = None
            smartvector_range_key = None
            vectorgenerator_pair = None
            kronos_frames = None
            motion3d_camera_data = None
            reference_paint = None
            inpaint_samples = None
            contactsheet_sequence = None
            timewarp_weights = None
            temporal_kinds = ("TimeBlur", "TimeEcho", "MotionBlur2D", "MotionBlur3D", "TimeWarp")
            transform_blur = kind == "Transform" and bool(params.get("motionblur", 0))
            if (kind in temporal_kinds or transform_blur) and not node["disabled"]:
                source_key = node["inputs"]["image"]
                if kind == "TimeWarp":
                    # "lookup" is resolved like any other animated param above (see
                    # `_resolve_params`): a curve on it already gives the fractional value at
                    # this outer `frame`. With no curve at all the stored constant means nothing
                    # -- Nuke's own default identity ("input frame == output frame") is what a
                    # bare TimeWarp plays as, so that is what an un-keyed "lookup" falls back to.
                    has_lookup_curve = bool(node_curves and "lookup" in node_curves)
                    target_frame = float(params["lookup"]) if has_lookup_curve else float(frame)
                    lookup_filter = params.get("lookup_filter", "blend")
                    if lookup_filter == "none" or target_frame.is_integer():
                        # "none": the fractional frame is passed straight into the nested call,
                        # exactly like TimeBlur's own shutter subframes -- an animated upstream
                        # source is sampled at that exact fractional position.
                        sample_frames, weights = [target_frame], [1.0]
                    else:
                        lo = math.floor(target_frame)
                        frac = target_frame - lo
                        sample_frames, weights = [lo, lo + 1], [1.0 - frac, frac]
                    samples = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=sample_frame,
                                                     tier=tier, typed=True, return_digest=True)
                               for sample_frame in sample_frames]
                    temporal_samples = samples
                    timewarp_weights = weights
                    fingerprint = ["time-warp", target_frame, lookup_filter, *(d for _, d in samples)]
                elif kind in ("TimeBlur", "MotionBlur", "MotionBlur2D", "MotionBlur3D") or transform_blur:
                    count = max(1, min(256, int(params["divisions"] if kind == "TimeBlur" else params["samples"])))
                    shutter = float(params["shutter"])
                    offset = params["shutter_offset"]
                    if offset == "start":
                        low, high = frame - shutter, frame
                    elif offset == "end":
                        low, high = frame, frame + shutter
                    elif offset == "custom":
                        low, high = frame + float(params["custom_offset"]) - shutter / 2, frame + float(params["custom_offset"]) + shutter / 2
                    else:
                        low, high = frame - shutter / 2, frame + shutter / 2
                    sample_frames = [low + (i + 0.5) * (high - low) / count for i in range(count)]
                    if transform_blur:
                        samples = []
                        for sample_frame in sample_frames:
                            sample_doc = copy.deepcopy(doc)
                            sample_doc["nodes"][key]["params"]["motionblur"] = 0
                            (sample_doc.get("animation", {}).get("curves", {}).get(key, {})
                             .pop("motionblur", None))
                            sample_doc.get("expressions", {}).get(key, {}).pop("motionblur", None)
                            samples.append(self.evaluate_raster(sample_doc, key, cancel=cancel,
                                                                frame=sample_frame, tier=tier,
                                                                typed=True, return_digest=True))
                    else:
                        # `cache_fractional` is TimeBlur-only (docstring on `evaluate_raster`):
                        # its shutter subframes are the ones the tile executor re-asks for, tile
                        # after tile and compose after compose, at the *same* outer frame.
                        samples = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=sample_frame,
                                                         tier=tier, typed=True, return_digest=True,
                                                         cache_fractional=(kind == "TimeBlur"))
                                   for sample_frame in sample_frames]
                    temporal_samples = samples
                    fingerprint = ["time-blur", *(d for _, d in samples)]
                    if kind == "MotionBlur3D" and node["inputs"].get("camera") is not None:
                        camera_key = node["inputs"]["camera"]
                        cameras = [values[camera_key]]
                        camera_results = [self.evaluate_raster(doc, camera_key, cancel=cancel, frame=f,
                            tier=tier, typed=True, return_digest=True) for f in (low, high)]
                        cameras.extend(result for result, _ in camera_results)
                        depth_key = node["inputs"].get("depth")
                        depth_digest = (self.evaluate_raster(doc, depth_key, cancel=cancel, frame=frame,
                            tier=tier, typed=True, return_digest=True)[1] if depth_key else None)
                        motion3d_camera_data = (cameras[0], cameras[1], cameras[2], depth_key)
                        fingerprint.extend(["camera-depth-motion", *(digest for _, digest in camera_results), depth_digest])
                else:
                    count = max(1, min(256, int(params["frames"])))
                    samples = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=frame - age,
                                                     tier=tier, typed=True, return_digest=True)
                               for age in range(count)]
                    temporal_samples = samples
                    fingerprint = ["time-echo", params["method"], params["falloff"], *(d for _, d in samples)]
            if kind == "VectorGenerator" and not node["disabled"]:
                source_key = node["inputs"]["image"]
                lo, hi = int(doc.get("time", {}).get("first", frame)), int(doc.get("time", {}).get("last", frame + 1))
                f0 = max(lo, min(hi, int(math.floor(frame))))
                f1 = max(lo, min(hi, f0 + 1))
                vectorgenerator_pair = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=f,
                    tier=tier, typed=True, return_digest=True) for f in (f0, f1)]
                fingerprint = ["vector-pair", f0, f1, *(d for _, d in vectorgenerator_pair)]
            if kind in ("Kronos", "OFlow") and not node["disabled"]:
                source_key = node["inputs"]["image"]
                if kind == "OFlow":
                    target_frame = (float(params["frame"]) if params["timing"] == "frame" else
                                    float(params["input_start"]) + (float(frame) - float(params["output_start"])) * float(params["speed"]))
                    lo = max(int(math.floor(float(params["input_start"]))), int(doc.get("time", {}).get("first", math.floor(target_frame))))
                    hi = min(int(math.ceil(float(params["input_end"]))), int(doc.get("time", {}).get("last", math.ceil(target_frame))))
                    if hi < lo:
                        raise ValueError("OFlow: input_end must be at or after input_start")
                    target_frame = min(float(hi), max(float(lo), target_frame))
                    count = max(1, min(64, int(params["shutter_samples"])))
                    shutter_times = ([target_frame] if count == 1 or float(params["shutter_time"]) == 0 else
                        np.linspace(target_frame - float(params["shutter_time"])/2,
                                    target_frame + float(params["shutter_time"])/2, count).tolist())
                    interpolation = "motion"
                else:
                    target_frame = float(params["frame"]) if float(params["frame"]) >= 0 else float(frame) * float(params["speed"])
                    lo, hi = int(doc.get("time", {}).get("first", math.floor(target_frame))), int(doc.get("time", {}).get("last", math.ceil(target_frame)))
                    target_frame = min(float(hi), max(float(lo), target_frame))
                    count = int(params["shutter_samples"]) if params["interpolation"] == "motion" else 1
                    shutter_times = [target_frame] if count == 1 else np.linspace(target_frame - abs(float(params["speed"])) / 2,
                        target_frame + abs(float(params["speed"])) / 2, count).tolist()
                    interpolation = params["interpolation"]
                pairs = []
                for sample_time in shutter_times:
                    q = min(float(hi), max(float(lo), float(sample_time)))
                    f0, f1 = int(math.floor(q)), int(math.ceil(q))
                    pair = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=f, tier=tier,
                        typed=True, return_digest=True) for f in (f0, f1)]
                    pairs.append((q, pair[0], pair[1]))
                kronos_frames = (target_frame, pairs)
                fingerprint = [kind.lower(), target_frame, interpolation,
                               *(d for _, a, b in pairs for d in (a[1], b[1]))]
            if kind == "MotionBlur" and not node["disabled"]:
                shutter = float(params["shutter"])
                if params["shutter_offset"] == "start": low, high = frame, frame + shutter
                elif params["shutter_offset"] == "end": low, high = frame - shutter, frame
                elif params["shutter_offset"] == "custom": low, high = frame + float(params["custom_offset"]) - shutter/2, frame + float(params["custom_offset"]) + shutter/2
                else: low, high = frame - shutter/2, frame + shutter/2
                source_key = node["inputs"]["image"]
                temporal_samples = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=f, tier=tier,
                    typed=True, return_digest=True) for f in (low, high)]
                fingerprint = ["motion-flow-blur", *(d for _,d in temporal_samples)]
            if kind == "SmartVector" and not node["disabled"]:
                source_key = node["inputs"]["image"]
                first = max(int(params["frame_start"]), int(doc.get("time", {}).get("first", params["frame_start"])))
                last = min(int(params["frame_end"]), int(doc.get("time", {}).get("last", params["frame_end"])))
                if last < first: raise ValueError("SmartVector: frame_end must be at or after frame_start")
                if not first <= int(params["reference_frame"]) <= last:
                    raise ValueError("SmartVector: reference_frame must be inside the analyzed range")
                sequence = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=f, tier=tier,
                                                  typed=True, return_digest=True)
                            for f in range(first, last + 1)]
                smartvector_sequence = sequence
                smartvector_first, smartvector_last = first, last
                smartvector_range_key = hashlib.sha256(json.dumps(
                    ["smartvector-range", key, params, first, last, tier,
                     *(digest for _, digest in sequence)], sort_keys=True).encode()).hexdigest()
                fingerprint = ["smartvector", first, last, int(params["reference_frame"]),
                               round(float(frame), 6), *(digest for _, digest in sequence)]
            if kind in ("VectorDistort", "VectorCornerPin") and not node["disabled"]:
                source_key = node["inputs"].get("image")
                if source_key is not None:
                    reference_paint = self.evaluate_raster(doc, source_key, cancel=cancel,
                        frame=float(params["reference_frame"]), tier=tier, typed=True, return_digest=True)
                    fingerprint = [kind.lower(), reference_paint[1], round(float(frame), 6)]
            if kind == "Inpaint" and not node["disabled"]:
                source_key = node["inputs"].get("image")
                radius = max(1, int(params["temporal_frames"]) // 2)
                lo, hi = int(doc.get("time", {}).get("first", frame-radius)), int(doc.get("time", {}).get("last", frame+radius))
                inpaint_samples = [self.evaluate_raster(doc, source_key, cancel=cancel,
                    frame=max(lo, min(hi, int(round(frame)) + offset)), tier=tier, typed=True, return_digest=True)
                    for offset in range(-radius, radius+1) if offset != 0]
                fingerprint = ["inpaint", params["fill_method"], params.get("flow_backend", "auto"),
                               *(d for _, d in inpaint_samples)]
            if kind == "ContactSheet" and params.get("splitinputs") and not node["disabled"]:
                source_key = node["inputs"].get("clip0")
                if source_key is not None:
                    start, end = int(params["startframe"]), int(params["endframe"])
                    if end < start:
                        end = start
                    contactsheet_sequence = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=f,
                        tier=tier, typed=True, return_digest=True) for f in range(start, end + 1)]
                    fingerprint = ["contactsheet-split", start, end, *(d for _, d in contactsheet_sequence)]
            if kind == "TimeDissolve" and not node["disabled"]:
                first, last = int(params["in"]), int(params["out"])
                t = 1.0 if last <= first and frame >= last else 0.0 if last <= first else min(1.0, max(0.0, (frame - first) / (last - first)))
                ease = params["ease"]
                if ease == "smooth":
                    t = t * t * (3.0 - 2.0 * t)
                elif ease == "animation curve":
                    curves = doc.get("animation", {}).get("curves", {}).get(key, {})
                    t = float(params["which"]) if "which" in curves else t
                params["which"] = t
                fingerprint = ["time-dissolve", round(t, 9)]
            if kind == "Denoise" and params.get("temporal") and not node["disabled"]:
                source_key = node["inputs"].get("image")
                temporal_samples = [self.evaluate_raster(doc, source_key, cancel=cancel, frame=sample_frame,
                                                         tier=tier, typed=True, return_digest=True)
                                    for sample_frame in (max(int(doc.get("time", {}).get("first", frame - 1)), frame - 1),
                                                        min(int(doc.get("time", {}).get("last", frame + 1)), frame + 1))]
            if kind in _TIME_REMAP_KINDS and not node["disabled"]:
                # The fingerprint is the *nested* call's own digest, not this walk's
                # `hashes[source]` (which was computed at the wrong, outer `frame`): a still keeps
                # one cache entry across every effective frame it is asked for, exactly like a
                # Read's fingerprint above, and a curve edited on a keyframe the outer walk never
                # visits at `frame` is still picked up, because the nested call resolves the
                # source's own curves at `effective_frame`, not at `frame`.
                if kind == "AppendClip":
                    # Up to two nested evaluations (two inside a dissolve), each of a different
                    # clip at its own frame; the blend weight is a function of `frame`, so it joins
                    # the fingerprint alongside the nested digests.
                    plan = _append_clip_plan(nodes, node, params, frame)
                    parts = [self.evaluate_raster(doc, target=node["inputs"][slot], cancel=cancel,
                                                  frame=source_frame, tier=tier, typed=True,
                                                  return_digest=True)
                             for slot, source_frame, _ in plan]
                    remap_raster = parts[0][0]
                    fingerprint = ["time_remap", *(digest for _, digest in parts)]
                    if len(parts) == 2:
                        weight = plan[1][2]
                        a, b = parts[0][0], parts[1][0]
                        if a.pixels.shape != b.pixels.shape or a.data != b.data or a.display != b.display:
                            raise ValueError(f"{node['name']}: dissolving clips need matching formats")
                        remap_raster = Raster((a.pixels * np.float32(1.0 - weight)
                                               + b.pixels * np.float32(weight)), a.data, a.display)
                        fingerprint.append(round(weight, 9))
                else:
                    black = False
                    if kind in ("TimeClip", "FrameRange"):
                        effective_frame, black = _time_clip_frame(kind, params, frame)
                    else:
                        effective_frame = _time_remap_frame(kind, params, frame)
                    source_key = node["inputs"][_SPECS[kind]["inputs"][0]]
                    remap_raster, remap_digest = self.evaluate_raster(
                        doc, target=source_key, cancel=cancel, frame=effective_frame, tier=tier,
                        typed=True, return_digest=True)
                    fingerprint = ["time_remap", remap_digest]
                    if black:
                        # Outside the range with before/after "black": a transparent frame at the
                        # size of the nearest in-range frame (`_range_frame` returned that one).
                        remap_raster = Raster(np.zeros_like(remap_raster.pixels), remap_raster.data,
                                              remap_raster.display, meta=remap_raster.meta)
                        fingerprint.append("black")
            # The tier is folded in explicitly rather than left implicit in the scaled parameters:
            # a Grade has no pixel units, so its parameters are identical at every tier while its
            # pixels are not. Without this, a tier 4 result would satisfy a tier 1 request (C1).
            # `data` joins the digest explicitly: a Roto's shapes are not in params, so without
            # this an edited shape would serve the previous matte out of cache. A Tracker needs no
            # term here because its solve already landed in params above.
            #
            # A time-remapping node folds in `fingerprint` (the nested call's digest at
            # `effective_frame`) instead of `hashes[source]`: `hashes[source]` was computed by
            # this walk at the outer `frame`, which is *not* the frame this node's pixels actually
            # come from, and an animated source would make it churn on every outer frame even
            # when `effective_frame` — and so the actual result — does not change (the FrameHold
            # cache-reuse case docs/TIME_MODEL.md and this lane's own tests require).
            source_hashes = ([] if kind in _TIME_REMAP_KINDS + ("TimeBlur", "TimeEcho", "MotionBlur", "MotionBlur2D", "MotionBlur3D", "TimeWarp") and not node["disabled"]
                             else [hashes[s] for s in sources if s is not None])
            if kind == "RotoPaint" and not node["disabled"]:
                input_key = node["inputs"].get("image")
                if input_key is not None:
                    source_frames = sorted({(max(int(doc["time"]["first"]), int(frame) - 1) if item["source_frame"] == "relative"
                                             else int(item["source_frame"]))
                                            for item in (data or [])
                                            if item.get("kind") == "stroke" and item.get("tool") == "clone"})
                    if source_frames:
                        clone_fingerprints = []
                        for source_frame in source_frames:
                            clone_raster, clone_digest = self.evaluate_raster(
                                doc, target=input_key, cancel=cancel, frame=source_frame, tier=tier,
                                typed=True, return_digest=True)
                            clone_sources[source_frame] = clone_raster
                            clone_fingerprints.append([source_frame, clone_digest])
                        fingerprint = ["clone_sources", clone_fingerprints]
            if temporal_samples is not None:
                fingerprint = [kind.lower(), *(sample_digest for _, sample_digest in temporal_samples)]
            digest = hashlib.sha256(json.dumps([kind, params, node["disabled"], source_hashes,
                                                node.get("input_outputs", {}), fingerprint, tier, data], sort_keys=True).encode()).hexdigest()
            hashes[key] = digest
            # `pixels`, not `frame`: in this module "frame" now means a position in time, and the
            # loop must not clobber the timeline frame that later Reads still need.
            if (not fractional_frame or cache_fractional) and digest in self.cache and key not in getattr(self, "_lut_roots", {}):
                self.hits += 1
                raster = self.cache.pop(digest)
                self.cache[digest] = raster
            else:
                self.misses += 1
                # A memory miss consults the disk tier before recomputing. A hit there repopulates
                # memory, so the second read of a spilled result is a memory hit again (C4).
                spilled = None if fractional_frame or key in getattr(self, "_lut_roots", {}) else self.disk.get_raster(digest)
                if spilled is not None:
                    self.disk_hits += 1
                    self._store(digest, spilled)
                    values[key] = spilled
                    continue
                if kind == "SmartVector" and smartvector_sequence is not None:
                    ix = min(smartvector_last, max(smartvector_first, int(round(frame)))) - smartvector_first
                    range_cache = self.disk.get_raster(smartvector_range_key)
                    if range_cache is None:
                        from .flow_nodes import accumulated_vectors
                        from .opticalflow import _sample
                        seq = [r.pixels for r, _ in smartvector_sequence]
                        ref_ix = int(params["reference_frame"]) - smartvector_first
                        fw, bw = accumulated_vectors(seq, ref_ix, vector_detail=int(params["vector_detail"]),
                            smoothness=float(params["smoothness"]), reanchor_interval=int(params["reanchor_interval"]),
                            backend=params.get("flow_backend", "auto"))
                        hh, ww = seq[ref_ix].shape[:2]
                        yy, xx = np.mgrid[:hh, :ww].astype(np.float32)
                        range_layers = {}
                        for frame_index, (forward, backward) in enumerate(zip(fw, bw)):
                            reverse_at = _sample(backward, xx + forward[..., 0], yy + forward[..., 1])
                            error = np.linalg.norm(forward + reverse_at, axis=2)
                            occluded = error > (.5 + .01 * np.linalg.norm(forward, axis=2))
                            for direction, field in (("forward", forward), ("backward", backward)):
                                rgba = np.zeros((hh, ww, 4), np.float32)
                                rgba[..., :2] = field * np.float32(tier); rgba[..., 2] = occluded; rgba[..., 3] = 1
                                range_layers[f"frame.{frame_index}.{direction}"] = Raster.of(
                                    rgba, smartvector_sequence[ref_ix][0].display)
                        reference = smartvector_sequence[ref_ix][0]
                        range_cache = Raster(reference.pixels, reference.data, reference.display,
                                             range_layers, reference.meta)
                        self.disk.put_raster(smartvector_range_key, range_cache)
                    smartvector_layers = {
                        "smartvector.forward": range_cache.layers[f"frame.{ix}.forward"],
                        "smartvector.backward": range_cache.layers[f"frame.{ix}.backward"]}
                # Build the inputs list in declared slot order (required then optional) so kernels
                # pick up `image` first and `mask` second. None for optional slots becomes None.
                if node["disabled"]:
                    # Only the passed-through input was evaluated; the other slots have no value.
                    # A Draw node's bypass slot ("image") is itself optional, so a disabled Ramp/
                    # Radial/Rectangle/Noise/Text with nothing wired into it has no upstream value
                    # to look up at all -- it passes a transparent frame at its own declared format
                    # instead, exactly as an unwired bypass on any other optional-image slot would.
                    if sources and sources[0] is not None:
                        raster = values[sources[0]]
                    elif kind in _DRAW_KINDS:
                        raster = Raster.of(np.zeros((int(params["height"]), int(params["width"]), 4),
                                                     np.float32))
                    elif sources and sources[0] is None:
                        raise ValueError(f"{node['name']}: connect at least one clip input")
                    else:
                        raster = values[sources[0]]
                elif key in getattr(self, "_lut_roots", {}):
                    raster = self._lut_roots[key]
                elif kind in _TIME_REMAP_KINDS:
                    raster = remap_raster
                elif kind == "VectorGenerator" and vectorgenerator_pair is not None:
                    from .opticalflow import flow_pair
                    a, b = (r for r, _ in vectorgenerator_pair)
                    if a.pixels.shape != b.pixels.shape:
                        raise ValueError("VectorGenerator: adjacent frames must have matching formats")
                    fw, bw, occ = flow_pair(a.pixels, b.pixels, vector_detail=int(params["vector_detail"]),
                                            smoothness=float(params["smoothness"]), flow_on=params["flow_on"],
                                            backend=params.get("flow_backend", "auto"))
                    layers = {}
                    for name, field in (("forward", fw), ("backward", bw)):
                        rgba = np.zeros((*field.shape[:2], 4), np.float32)
                        rgba[..., :2] = field * np.float32(tier); rgba[..., 3] = 1
                        layers[f"vector.{name}"] = Raster.of(rgba, a.display)
                    occ_rgba = np.zeros((*occ.shape,4), np.float32); occ_rgba[...,0] = occ; occ_rgba[...,3] = 1
                    layers["vector.occlusion"] = Raster.of(occ_rgba, a.display)
                    src = values[node["inputs"]["image"]]
                    raster = Raster(src.pixels, src.data, src.display, {**(src.layers or {}), **layers}, src.meta)
                elif kind in ("Kronos", "OFlow") and kronos_frames is not None:
                    from .flow_nodes import warp_by_flow
                    from .opticalflow import flow_pair
                    target_frame, pairs = kronos_frames
                    rendered = []
                    for sample_time, (a, _), (b, _) in pairs:
                        alpha = np.float32(sample_time - math.floor(sample_time))
                        if a.display != b.display:
                            raise ValueError("Kronos: sampled frames must have matching display windows")
                        output_data = a.data.union(b.data)
                        ap, bp = a.fit(output_data), b.fit(output_data)
                        if sample_time == math.floor(sample_time): pixels = ap.copy()
                        elif (params["interpolation"] if kind == "Kronos" else "motion") == "frame": pixels = ap * (1-alpha) + bp * alpha
                        else:
                            fw, bw, occ = flow_pair(ap, bp,
                                vector_detail=int(params.get("vector_detail", 4)),
                                smoothness=float(params.get("smoothness", 1.0)),
                                backend=params.get("flow_backend", "auto"))
                            wa = warp_by_flow(ap, fw * alpha)
                            wb = warp_by_flow(bp, bw * (1-alpha))
                            pixels = wa * (1-alpha) + wb * alpha
                            pixels[occ] = wb[occ]
                        rendered.append((pixels, output_data, a))
                    pixels0, output_data, a = rendered[0]
                    aligned = [pixels if data == output_data else Raster(pixels, data, a.display).fit(output_data)
                               for pixels, data, _ in rendered]
                    pixels = aligned[0] if len(aligned) == 1 else np.mean(np.stack(aligned), axis=0, dtype=np.float32)
                    raster = Raster(pixels.astype(np.float32), output_data, a.display, a.layers, a.meta)
                elif kind == "VectorToMotion":
                    source = values[node["inputs"]["image"]]
                    from .flow_nodes import vector_layers_to_motion
                    layers = vector_layers_to_motion(source.layers, params["forward_layer"], params["backward_layer"])
                    raster = Raster(source.pixels, source.data, source.display, layers, source.meta)
                elif kind == "SmartVector" and smartvector_layers is not None:
                    source = values[node["inputs"]["image"]]
                    raster = Raster(source.pixels, source.data, source.display,
                                    {**(source.layers or {}), **smartvector_layers}, source.meta)
                elif kind in ("VectorDistort", "VectorCornerPin") and reference_paint is not None:
                    from .flow_nodes import homography_from_corners, warp_by_flow, warp_by_homography
                    base, _ = reference_paint
                    vectors_key = node["inputs"].get("vectors")
                    vectors = values.get(vectors_key) if vectors_key else None
                    layer_name = "smartvector.forward" if frame >= float(params["reference_frame"]) else "smartvector.backward"
                    vector_layer = None if vectors is None else (vectors.layers or {}).get(layer_name)
                    if vector_layer is None:
                        raise ValueError(f"{kind}: connect a SmartVector node to vectors")
                    field = vector_layer.pixels[..., :2] / np.float32(tier)
                    if frame < float(params["reference_frame"]):
                        field = -field   # backward layer maps current -> reference; invert for ref -> current.
                    if field.shape[:2] != base.pixels.shape[:2]:
                        raise ValueError(f"{kind}: vector and paint formats must match")
                    gpu_warp = False
                    vector_node = nodes.get(vectors_key) if vectors_key else None
                    if (vector_node is not None and vector_node.get("type") == "SmartVector"
                            and not vector_node.get("disabled")):
                        vector_params = _resolve_params(vector_node,
                            doc.get("animation", {}).get("curves", {}).get(vectors_key), frame,
                            _SPECS["SmartVector"]["params"], _LIMITS)
                        backend = vector_params.get("flow_backend", "auto")
                        if backend == "gpu":
                            gpu_warp = True
                        elif backend == "auto" and vector_params.get("flow_on", "luminance") == "luminance":
                            from . import gpu3d
                            gpu_warp = gpu3d.available()
                    if kind == "VectorCornerPin":
                        h, w = base.pixels.shape[:2]
                        points = np.asarray([[params[f"corner{i}_x"], params[f"corner{i}_y"]]
                                             for i in range(1, 5)], np.float32)
                        if np.allclose(points, [[0, 0], [100/tier, 0], [100/tier, 100/tier], [0, 100/tier]]):
                            points = np.asarray([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], np.float32)
                        points[:, 0] = np.clip(points[:, 0], 0, w-1); points[:, 1] = np.clip(points[:, 1], 0, h-1)
                        yy, xx = np.mgrid[:h, :w].astype(np.float32)
                        displacement = np.stack([__import__("nodebased.opticalflow", fromlist=["_sample"])._sample(
                            field[..., axis], points[:, 0], points[:, 1]) for axis in range(2)], axis=1)
                        matrix = homography_from_corners(points, points + displacement)
                        if gpu_warp:
                            from .opticalflow_gpu import warp_by_homography_gpu
                            warped = warp_by_homography_gpu(base.pixels, matrix)
                        else:
                            warped = warp_by_homography(base.pixels, matrix)
                    else:
                        if gpu_warp:
                            from .opticalflow_gpu import warp_by_flow_gpu
                            warped = warp_by_flow_gpu(base.pixels, field)
                            if float(params["blur_size"]) > 0:
                                # Preserve the reference blur kernel after GPU resampling.
                                from .flow_nodes import warp_by_flow as _cpu_blur
                                unblurred = warped.copy()
                                warped = _cpu_blur(unblurred, np.zeros_like(field),
                                                   blur_size=float(params["blur_size"]))
                        else:
                            warped = warp_by_flow(base.pixels, field, blur_size=float(params["blur_size"]))
                    if kind == "VectorDistort":
                        fade = int(params["fade_frames"])
                        amount = 1.0 if fade <= 0 else max(0.0, 1.0 - abs(float(frame)-float(params["reference_frame"])) / fade)
                        warped *= np.float32(amount)
                        mask = values.get(node["inputs"].get("mask"))
                        warped = self._apply_mask_mix(base.pixels, warped,
                            None if mask is None else mask.fit(base.data), float(params["mix"]))
                    else:
                        mask = values.get(node["inputs"].get("mask"))
                        warped = self._apply_mask_mix(base.pixels, warped,
                            None if mask is None else mask.fit(base.data), float(params["mix"]))
                    raster = Raster(warped.astype(np.float32), base.data, base.display, base.layers, base.meta)
                elif kind == "MotionBlur" and temporal_samples is not None:
                    from .opticalflow import flow_pair
                    (a, _), (b, _) = temporal_samples
                    if a.display != b.display:
                        raise ValueError("MotionBlur: shutter samples must have matching display windows")
                    output_data = a.data.union(b.data)
                    ap, bp = a.fit(output_data), b.fit(output_data)
                    flow = self._motion_flow(values[node["inputs"]["image"]], ap, bp,
                                             output_data, params)
                    pblur = {"vector_scale": 1.0, "max_length": 100.0, "vector_offset": 0.0,
                             "vector_method": "forward", "vector_alpha": "none", "samples": int(params["samples"])}
                    blurred = self._vector_blur(ap, flow[...,0], flow[...,1], pblur)
                    mask = values.get(node["inputs"].get("mask"))
                    raster = Raster(self._apply_mask_mix(a.pixels, blurred,
                        None if mask is None else mask.fit(output_data), float(params["mix"])), output_data, a.display, a.layers, a.meta)
                elif kind == "TimeWarp" and temporal_samples is not None:
                    sampled = [r for r, _ in temporal_samples]
                    reference = sampled[0]
                    if len(sampled) > 1 and sampled[1].display != reference.display:
                        raise ValueError("TimeWarp: sampled frames must have matching display windows")
                    output_data = reference.data
                    for item in sampled[1:]:
                        output_data = output_data.union(item.data)
                    frames_fit = [r.fit(output_data) for r in sampled]
                    if len(frames_fit) == 1:
                        pixels = frames_fit[0].copy()
                    else:
                        w0, w1 = timewarp_weights
                        pixels = frames_fit[0] * np.float32(w0) + frames_fit[1] * np.float32(w1)
                    raster = Raster(pixels.astype(np.float32), output_data, reference.display,
                                    reference.layers, reference.meta)
                elif kind in ("TimeBlur", "TimeEcho", "MotionBlur2D", "MotionBlur3D") or transform_blur:
                    sampled = [r for r, _ in temporal_samples]
                    reference = sampled[0]
                    if any(r.display != reference.display for r in sampled):
                        raise ValueError(f"{kind}: sampled frames must have matching display windows")
                    output_data = reference.data
                    for item in sampled[1:]:
                        output_data = output_data.union(item.data)
                    frames = [r.fit(output_data) for r in sampled]
                    if kind in ("TimeBlur", "MotionBlur", "MotionBlur2D", "MotionBlur3D") or transform_blur:
                        pixels = (frames[0].copy() if all(np.array_equal(frames[0], f) for f in frames[1:])
                                  else np.mean(np.stack(frames), axis=0, dtype=np.float32))
                    else:
                        falloff = min(1.0, max(0.0, float(params["falloff"])))
                        weights = np.asarray([falloff ** i for i in range(len(frames))], dtype=np.float32)
                        weighted = [item * weights[i] for i, item in enumerate(frames)]
                        if params["method"] == "max":
                            pixels = np.maximum.reduce(weighted)
                        elif params["method"] == "plus":
                            pixels = np.sum(weighted, axis=0, dtype=np.float32)
                        else:
                            pixels = np.sum(weighted, axis=0, dtype=np.float32) / max(float(np.sum(weights)), 1e-12)
                    if kind == "MotionBlur3D" and motion3d_camera_data is not None:
                        from . import scene3d
                        current_camera, camera_start, camera_end, depth_key = motion3d_camera_data
                        base_raster = values[node["inputs"]["image"]]
                        depth_raster = values.get(depth_key) if depth_key else None
                        if depth_raster is None:
                            depth_raster = (base_raster.layers or {}).get("depth.Z")
                        if depth_raster is None:
                            raise ValueError("MotionBlur3D: connect a depth pass and a camera")
                        if not isinstance(depth_raster, Raster):
                            raise ValueError("MotionBlur3D: depth input must be an image or depth layer")
                        if not all(isinstance(camera, scene3d.Camera) for camera in
                                   (current_camera, camera_start, camera_end)):
                            raise ValueError("MotionBlur3D: camera input must evaluate to a camera")
                        depth_values = depth_raster.fit(output_data)[..., 0]
                        flow = self._camera_depth_flow(depth_values, current_camera, camera_start,
                                                       camera_end, reference.display, output_data)
                        blur_params = {"vector_scale": 1.0, "max_length": 0.0,
                                       "vector_offset": -0.5, "vector_method": "forward",
                                       "vector_sampling": "source", "vector_alpha": "none",
                                       "samples": int(params["samples"])}
                        pixels = self._vector_blur(base_raster.fit(output_data),
                                                   flow[..., 0], flow[..., 1], blur_params)
                    raster = Raster(pixels.astype(np.float32), output_data, reference.display, reference.layers, reference.meta)
                    if kind in ("MotionBlur2D", "MotionBlur3D") or transform_blur:
                        mask_slot = "mask"
                        mask_id = node["inputs"].get(mask_slot)
                        mask = values[mask_id] if mask_id is not None else None
                        base = values[node["inputs"]["image"]].fit(output_data)
                        raster = Raster(self._apply_mask_mix(base, raster.fit(output_data),
                                             None if mask is None else mask.fit(output_data), params["mix"]),
                                        output_data, reference.display, reference.layers, reference.meta)
                else:
                    slot_sources = [node["inputs"][s] for s in _SPECS[kind]["inputs"]]
                    slot_sources.extend(node["inputs"].get(s) for s in _SPECS[kind].get("optional_inputs", []))
                    selected_outputs = node.get("input_outputs", {})
                    images = []
                    for slot, source in zip(_SPECS[kind]["inputs"] + _SPECS[kind].get("optional_inputs", []), slot_sources):
                        image = values[source] if source is not None else None
                        if image is not None and selected_outputs.get(slot) == "out2":
                            image = (image.layers or {}).get("out2")
                            if image is None:
                                raise ValueError(f"{nodes[source]['name']}: out2 has no image")
                        images.append(image)
                    if kind == "ContactSheet" and contactsheet_sequence is not None:
                        # Split-inputs mode (step D2 finish): the grid is filled from one input's
                        # frame range instead of the separate clip0..clip31 slots, so the images the
                        # placement code in _windowed_kernel sees are the frame sequence, and each
                        # cell's own frame number stands in for a per-clip name.
                        images = [r for r, _ in contactsheet_sequence]
                        start = int(params["startframe"])
                        params = dict(params, _contact_labels=[str(start + i) for i in range(len(images))])
                    if kind == "RotoPaint" and clone_sources:
                        enriched = []
                        for item in data or []:
                            if item.get("kind") == "stroke" and item.get("tool") == "clone":
                                clone_frame = (max(int(doc["time"]["first"]), int(frame)-1)
                                               if item["source_frame"] == "relative" else int(item["source_frame"]))
                                item = dict(item, _clone_sources=clone_sources, _clone_frame=clone_frame)
                            enriched.append(item)
                        data = enriched
                    if kind == "GridWarpTracker" and params.get("drive", "tracker") == "smartvector" and len(images) > 2 and images[2] is not None:
                        reference = int(params.get("reference_frame", 1))
                        layer_name = (params.get("forward_layer", "smartvector.forward")
                                      if frame >= reference else params.get("backward_layer", "smartvector.backward"))
                        if layer_name not in (images[2].layers or {}):
                            vector_key = node["inputs"].get("vectors")
                            direction = -1 if frame >= reference else 1
                            candidate = int(math.floor(frame))
                            lower = int(doc.get("time", {}).get("first", candidate))
                            upper = int(doc.get("time", {}).get("last", candidate))
                            while vector_key is not None and candidate != reference:
                                candidate = min(upper, max(lower, candidate + direction))
                                if candidate == reference:
                                    break
                                prior_vectors = self.evaluate_raster(doc, target=vector_key, cancel=cancel,
                                    frame=candidate, tier=tier, typed=True)
                                if layer_name in (prior_vectors.layers or {}):
                                    images[2] = prior_vectors
                                    break
                        fingerprint = ["gridwarp-tracker-smartvector", round(float(frame), 6), layer_name]
                    if kind == "GridWarpTracker" and params.get("drive") == "smartvector" and tier != 1 and len(images) > 2 and images[2] is not None:
                        vector_raster = images[2]
                        layers = dict(vector_raster.layers or {})
                        for name, layer in layers.items():
                            if name.startswith(("vector.", "smartvector.")):
                                pixels = layer.pixels.copy()
                                pixels[..., :2] /= np.float32(tier)
                                layers[name] = Raster(pixels, layer.data, layer.display, layer.layers, layer.meta)
                        images[2] = Raster(vector_raster.pixels, vector_raster.data, vector_raster.display,
                                           layers, vector_raster.meta)
                    if kind == "GridWarpTracker" and len(images) > 2 and images[2] is not None:
                        vectors_now = images[2]
                        vector_key = node["inputs"].get("vectors")
                        reference = int(params.get("reference_frame", 1))
                        tracker_points = (data or {}).get("tracker_points") if isinstance(data, dict) else None
                        if tracker_points:
                            sample_src = np.asarray(tracker_points[0], dtype=float)
                            pending = np.array([not self._gridwarp_vector_valid(point, vectors_now, params, frame)
                                                for point in sample_src], dtype=bool)
                        elif params.get("drive") == "smartvector":
                            from .warps import default_grid
                            points = np.asarray(default_grid(images[0].display.width, images[0].display.height,
                                                            params["rows"], params["columns"]), dtype=float).reshape(-1, 2)
                            pending = np.array([not self._gridwarp_vector_valid(point, vectors_now, params, frame)
                                                for point in points], dtype=bool)
                        else:
                            pending = np.zeros(0, dtype=bool)
                        if vector_key is not None and pending.any() and int(frame) != reference:
                            lower = int(doc.get("time", {}).get("first", int(frame)))
                            history_layers = dict(vectors_now.layers or {})
                            history_digests = []
                            track_history = (data or {}).get("tracker_history", []) if isinstance(data, dict) else []
                            track_positions = {int(candidate): positions for candidate, positions in track_history}
                            for candidate in range(int(frame) - 1, lower - 1, -1):
                                prior, prior_digest = self.evaluate_raster(
                                    doc, target=vector_key, cancel=cancel, frame=candidate, tier=tier,
                                    typed=True, return_digest=True)
                                history_digests.append([candidate, prior_digest])
                                for name, layer in (prior.layers or {}).items():
                                    if tier != 1 and name.startswith(("vector.", "smartvector.")):
                                        pixels = layer.pixels.copy()
                                        pixels[..., :2] /= np.float32(tier)
                                        layer = Raster(pixels, layer.data, layer.display, layer.layers, layer.meta)
                                    history_layers[f"gridwarp.history.{candidate}|{name}"] = layer
                                candidate_raster = self._gridwarp_history_raster(
                                    Raster(vectors_now.pixels, vectors_now.data, vectors_now.display,
                                           history_layers, vectors_now.meta), candidate)
                                if candidate_raster is None:
                                    continue
                                if tracker_points:
                                    positions = track_positions.get(candidate, [])
                                    for index in np.flatnonzero(pending):
                                        if index < len(positions) and positions[index] is not None and self._gridwarp_vector_valid(
                                                sample_src[index], candidate_raster, params, candidate):
                                            pending[index] = False
                                else:
                                    for index in np.flatnonzero(pending):
                                        if self._gridwarp_vector_valid(points[index], candidate_raster, params, candidate):
                                            pending[index] = False
                                if not pending.any():
                                    break
                            images[2] = Raster(vectors_now.pixels, vectors_now.data, vectors_now.display,
                                               history_layers, vectors_now.meta)
                            fingerprint = [fingerprint, "gridwarp_history", history_digests]
                    if temporal_samples is not None:
                        source = images[0]
                        prev, nxt = (r for r, _ in temporal_samples)
                        if prev.data != source.data or nxt.data != source.data:
                            raise ValueError("Denoise temporal frames must have matching data windows")
                        average = (prev.pixels + source.pixels + nxt.pixels) / np.float32(3.0)
                        denoised = self._bilateral(average, {"spatial_size": 2.0 + 5.0 * float(params.get("denoise_strength", 0.2)),
                                                             "colour_sigma": max(0.01, float(params.get("denoise_strength", 0.2)))})
                        mask = images[1] if len(images) > 1 else None
                        pixels = self._apply_mask_mix(source.pixels, denoised,
                                                      None if mask is None else mask.fit(source.data), params.get("mix", 1.0))
                        raster = Raster(pixels, source.data, source.display, source.layers, source.meta)
                    elif kind == "Inpaint" and inpaint_samples is not None:
                        from .flow_nodes import inpaint
                        source = images[0]
                        matte = images[1]
                        if matte is None:
                            raise ValueError("Inpaint: connect a matte")
                        m = matte.fit(source.data)
                        neighbours = [r.fit(source.data) for r, _ in inpaint_samples]
                        filled = inpaint(source.pixels, m[..., 3], neighbours,
                                         params["fill_method"], params.get("flow_backend", "auto"))
                        pixels = self._apply_mask_mix(source.pixels, filled, None, params["mix"])
                        raster = Raster(pixels.astype(np.float32), source.data, source.display,
                                        source.layers, source.meta)
                    elif kind == "Assert":
                        # `node["name"]` (the artist's own instance name, not the type) is only in
                        # scope in this per-node loop, which is why Assert is handled here rather
                        # than in the generic `_windowed_kernel` (a `@staticmethod` with no node).
                        from .ops2d_assert import evaluate_condition
                        source = images[0]
                        if not evaluate_condition(str(params["condition"]), source.pixels, frame):
                            raise ValueError(f'{node["name"]}: {params["message"]}')
                        raster = source
                    else:
                        raster = self._windowed_kernel(kind, params, images, frame, data, cancel=cancel,
                                                       progress=self)
                    if raster.meta is None and not any(raster is image for image in images):
                        raster.meta = _inherited_metadata(kind, images)
                if tier != 1 and kind in ("Read", "ReadBundle") and not node["disabled"]:
                    # A file cannot be decoded at a fraction of its size, so a Read is the one
                    # source that must decimate after the fact. Everything downstream of it still
                    # runs small, which is where the saving lives. Both windows move with the
                    # pixels; a data window left in full-resolution coordinates would place the
                    # overscan four times too far out at tier 4.
                    decimated = self._decimate(raster.pixels, tier)
                    layer_rasters = {}
                    for name, layer in (raster.layers or {}).items():
                        layer_pixels = self._decimate(layer.pixels, tier)
                        layer_data = scale_window(layer.data, tier, layer_pixels.shape[1], layer_pixels.shape[0])
                        layer_rasters[name] = Raster(layer_pixels, layer_data, layer.display.scaled(tier),
                                                     meta=layer.meta)
                    raster = Raster(decimated,
                                    scale_window(raster.data, tier, decimated.shape[1], decimated.shape[0]),
                                    raster.display.scaled(tier), layer_rasters, raster.meta)
                raster.pixels.flags.writeable = False
                if not fractional_frame or cache_fractional:
                    self._store(digest, raster)
            values[key] = raster
        result = values[target]
        if not isinstance(result, Raster) and not typed:
            raise ValueError(f"{nodes[target]['name']} outputs {OUTPUT_TYPES.get(nodes[target]['type'])}, "
                             "not an image; view a Render3D instead")
        if return_digest:
            return result, hashes[target]
        return result

    # --- bounding box ---------------------------------------------------------------------------
    #
    # `_kernel` stays a pure array function: it is the reference math and it is tested directly.
    # Everything about *where* an image lives lives here instead, in one place, so a new kernel
    # cannot accidentally invent its own window convention. See docs/BOUNDING_BOX.md.

    @staticmethod
    def _windowed_kernel(kind, p, inputs, frame=None, data=None, cancel=None, progress=None):
        """Run a kernel with its inputs aligned to the output's data window.

        Two rules decide every case:
          * Which rectangle does this node's output cover? (`_filter_window` for the filters;
            generators state their own; Merge takes the union of its inputs'.)
          * Are its inputs aligned into that rectangle before the array math runs? Always — no
            kernel ever sees two arrays that disagree about where their pixels are.

        `cancel`, when given, lets the three MASK_MIX_KINDS members named in docs/M1_GATE.md's
        "interactive cancellation" section (`Grade`, `ColorCorrect`, `Saturation`, measured there
        at 4K as ~140/160/290 ms each -- already over the 100 ms cancel budget on their own) be
        interrupted mid-kernel; see `_ROW_CHUNKED_MASK_MIX_KINDS` below.
        """
        from .core import DRAW_KINDS, MASK_MIX_KINDS, MERGE_LIKE_KINDS, METADATA_KINDS, UV_KINDS, WINDOW_KINDS

        if kind == "Read":
            return read_image_raster(**p, frame=frame)
        if kind == "GenerateLUT":
            return inputs[0]
        if kind == "Vectorfield":
            source = inputs[0]
            if source is None:
                raise ValueError("Vectorfield: connect an image")
            from . import lutio
            if not p["cube_path"]:
                raise ValueError("Vectorfield: choose a .cube or .3dl file")
            try:
                suffix = Path(p["cube_path"]).suffix.lower()
                if suffix == ".cube":
                    lut, shaper = lutio.read_cube(p["cube_path"]), None
                elif suffix == ".3dl":
                    lut, shaper = lutio.read_3dl(p["cube_path"])
                else:
                    raise ValueError("choose a .cube or .3dl LUT file")
                alpha = source.pixels[..., 3:4]
                straight = np.divide(source.pixels[..., :3], alpha, out=np.zeros_like(source.pixels[..., :3]), where=alpha != 0)
                working = lutio.convert_colorspace(straight, "ACEScg", p["colorspace_in"])
                if shaper is not None:
                    working = lutio._apply_shaper(working, shaper)
                mapped = lutio._sample(lut, working, p["interpolation"])
                mapped = lutio.convert_colorspace(mapped, p["colorspace_out"], "ACEScg")
                filtered = source.pixels.copy()
                filtered[..., :3] = mapped * alpha
                mask = inputs[1] if len(inputs) > 1 else None
                pixels = Evaluator._apply_mask_mix(source.pixels, filtered,
                    None if mask is None else mask.fit(source.data), p.get("mix", 1.0))
                return Raster(pixels, source.data, source.display, source.layers, source.meta)
            except (OSError, ValueError) as error:
                raise ValueError(f"Vectorfield: {error}") from None
        if kind == "ContactSheet":
            clips = [clip for clip in inputs if clip is not None]
            if not clips:
                raise ValueError("ContactSheet: connect at least one clip")
            width, height = max(1, int(p["width"])), max(1, int(p["height"]))
            rows, cols = max(1, int(p["rows"])), max(1, int(p["columns"]))
            gap = max(0, int(p["gap"]))
            roworder, colorder = p.get("roworder", "BottomTop"), p.get("colorder", "LeftRight")
            out = np.zeros((height, width, 4), np.float32)
            cell_w = max(1, (width - gap * (cols + 1)) // cols)
            cell_h = max(1, (height - gap * (rows + 1)) // rows)
            placed = clips[:rows * cols]
            center = bool(p.get("center", 0))
            for i, clip in enumerate(placed):
                row, col = Evaluator._contact_sheet_cell(i, cols, rows, roworder, colorder,
                                                          center=center, count=len(placed))
                x0, y0 = gap + col * (cell_w + gap), gap + row * (cell_h + gap)
                x1, y1 = min(width, x0 + cell_w), min(height, y0 + cell_h)
                src = clip.to_display()
                sh, sw = src.shape[:2]
                scale = (min((x1-x0)/sw, (y1-y0)/sh) if p["fit"] == "fit"
                         else max((x1-x0)/sw, (y1-y0)/sh))
                rw, rh = max(1, round(sw*scale)), max(1, round(sh*scale))
                sx = np.clip((np.arange(rw, dtype=np.float32)+.5)/scale-.5, 0, sw-1)
                sy = np.clip((np.arange(rh, dtype=np.float32)+.5)/scale-.5, 0, sh-1)
                xl, yl = np.floor(sx).astype(int), np.floor(sy).astype(int)
                xh, yh = np.minimum(xl+1, sw-1), np.minimum(yl+1, sh-1)
                fx, fy = (sx-xl)[None,:,None], (sy-yl)[:,None,None]
                a = src[yl[:,None], xl[None,:]]*(1-fx)+src[yl[:,None], xh[None,:]]*fx
                b = src[yh[:,None], xl[None,:]]*(1-fx)+src[yh[:,None], xh[None,:]]*fx
                resized = a*(1-fy)+b*fy
                if p["fit"] == "fill":
                    ox, oy = max(0,(rw-(x1-x0))//2), max(0,(rh-(y1-y0))//2)
                    resized = resized[oy:oy+(y1-y0), ox:ox+(x1-x0)]
                    dx, dy = x0, y0
                else:
                    dx, dy = x0+(x1-x0-rw)//2, y0+(y1-y0-rh)//2
                hh, ww = min(resized.shape[0], height-dy), min(resized.shape[1], width-dx)
                if hh > 0 and ww > 0:
                    out[dy:dy+hh, dx:dx+ww] = resized[:hh,:ww]
            if p["labels"] != "none":
                for i in range(len(placed)):
                    row, col = Evaluator._contact_sheet_cell(i, cols, rows, roworder, colorder,
                                                              center=center, count=len(placed))
                    label = (p.get("_contact_labels", [])[i] if p["labels"] == "name"
                             else str(int(frame)))
                    label_scale = max(1, min(3, cell_h//24))
                    _paint_contact_label(out, label, gap+col*(cell_w+gap)+4,
                                         gap+row*(cell_h+gap)+4, label_scale)
            return Raster.of(out)
        if kind == "Tile":
            source = inputs[0]
            if source is None:
                raise ValueError("Tile: connect an image")
            pixels = source.fit(source.data)
            height, width = pixels.shape[:2]
            rows, columns = int(p["rows"]), int(p["columns"])
            out = np.empty_like(pixels)
            for row in range(rows):
                y0, y1 = row * height // rows, (row + 1) * height // rows
                for column in range(columns):
                    x0, x1 = column * width // columns, (column + 1) * width // columns
                    # Bilinear resize at pixel centres (no new image dependency).
                    sx = np.clip((np.arange(x1-x0, dtype=np.float32) + .5) * width / (x1-x0) - .5, 0, width-1)
                    sy = np.clip((np.arange(y1-y0, dtype=np.float32) + .5) * height / (y1-y0) - .5, 0, height-1)
                    xlo, ylo = np.floor(sx).astype(int), np.floor(sy).astype(int)
                    xhi, yhi = np.minimum(xlo+1, width-1), np.minimum(ylo+1, height-1)
                    fx, fy = (sx-xlo)[None, :, None], (sy-ylo)[:, None, None]
                    top = pixels[ylo[:, None], xlo[None, :]] * (1-fx) + pixels[ylo[:, None], xhi[None, :]] * fx
                    bottom = pixels[yhi[:, None], xlo[None, :]] * (1-fx) + pixels[yhi[:, None], xhi[None, :]] * fx
                    tile = top * (1-fy) + bottom * fy
                    if p["mirror_x"] and column % 2:
                        tile = tile[:, ::-1]
                    if p["mirror_y"] and row % 2:
                        tile = tile[::-1]
                    out[y0:y1, x0:x1] = tile
            mixed = Evaluator._apply_mask_mix(pixels, out, mask=None, mix=p.get("mix", 1.0))
            return Raster(mixed, source.data, source.display, source.layers, source.meta)
        if kind == "ReadBundle":
            from . import bundle
            return bundle.read_bundle_raster(**p, frame=frame)
        if kind == "ConditionedRead":
            from .conditioned_read import read_conditioned_sequence
            conditioned = read_conditioned_sequence(p["path"], p["manifest"], p["scene_state"],
                                                    frame_offset=p.get("frame_offset", 0))
            match = next((item for item in conditioned if item.frame == int(frame)), None)
            if match is None:
                raise ValueError(f"ConditionedRead has no exported frame {frame}")
            return Raster(match.beauty, layers=match.layers)
        if kind == "Generate":
            source = inputs[0] if inputs else None
            if source is None:
                raise ValueError("Generate: connect a source plate image")
            # Provider execution is an explicit panel action. The node itself is a Write-like tap
            # so evaluating the graph never launches a model or changes the live plate.
            return source
        if kind in ("Constant", "Checker"):
            # A generated source defines the frame: data window and display window coincide.
            return Raster.of(Evaluator._kernel(kind, p, [], frame))
        if kind == "Roto":
            # Also a generator, and deliberately one: a Roto that inherited an input's format
            # could disagree with it about the data window and produce a matte that silently
            # fails to line up with the thing it is masking.
            return Raster.of(Evaluator._kernel(kind, p, [], frame, data))
        if kind in DRAW_KINDS:
            # A fourth kind of generator: still states its own format like Roto, but also takes an
            # optional "image" it composites the shape over, and an optional "mask" -- so unlike
            # Roto it is not exempt from the mask/mix contract every gated filter honours.
            width, height = int(p["width"]), int(p["height"])
            out = Region(0, 0, width, height)
            image, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if image is not None and image.display != out:
                raise ValueError(f"{kind} image input must match its own format in M0")
            if mask is not None and mask.display != out:
                raise ValueError(
                    f"Mask display window {mask.display} does not match {kind}'s own format {out}; "
                    "no silent resampling is performed")
            background = image.fit(out) if image is not None else np.zeros((height, width, 4), np.float32)
            shape = Evaluator._draw_shape(kind, p, 0, 0, width, height, frame)
            composited = Evaluator._composite_shape_over(shape, background)
            pixels = Evaluator._apply_mask_mix(background, composited,
                                               mask.fit(out) if mask is not None else None,
                                               p.get("mix", 1.0))
            return Raster.of(pixels)
        if kind == "Relight":
            bundle_raster = inputs[0]
            camera = inputs[1]  # Accepted for future use; v1 uses the render's baked camera response.
            layers = bundle_raster.layers
            if layers is None:
                raise ValueError("Relight: connect a Render3D node with Output set to 'Relight passes'")
            if "albedo" not in layers:
                raise ValueError("Relight: input bundle is missing the 'albedo' layer")
            if p["mix"] == 0:
                return bundle_raster.with_pixels(bundle_raster.pixels)
            # Ambient is NOT scaled by the Diffuse knob (docs/3D_ROADMAP.md "Design: relight
            # passes"): Diffuse/Specular scale only the per-light response sums, so setting either
            # to 0 turns off that kind of light contribution without also killing the ambient fill.
            ambient = np.empty_like(bundle_raster.pixels[..., :3])
            ambient[...] = (p["red"], p["green"], p["blue"])
            if "occlusion" in layers and p.get("use_intrinsics", "on") == "on":
                # A splat bundle carries per-pixel ambient visibility from the de-lit layer; the fill
                # light reaches less of a crease. Absent for mesh bundles, and off restores the old sum.
                ambient *= layers["occlusion"].pixels[..., :1]
            diffuse_sum = np.zeros_like(ambient)
            specular_total = np.zeros_like(ambient)
            for i, light in enumerate(inputs[2:10]):
                if light is None or light.intensity <= 0:
                    continue
                colour = np.asarray(light.color, dtype=np.float32) * light.intensity
                if f"diffuse_L{i}" in layers:
                    diffuse_sum += layers[f"diffuse_L{i}"].pixels[..., :3] * colour
                if f"specular_L{i}" in layers:
                    specular_total += layers[f"specular_L{i}"].pixels[..., :3] * colour
            # Splat bundles also carry the environment's diffuse and specular light and the traced mesh
            # reflections; `Environment` and `Reflections` rebalance them, `Specular` scales all specular.
            environment = p.get("environment", 1.0)
            if "environment_diffuse" in layers:
                diffuse_sum = diffuse_sum * p["diffuse"] + layers["environment_diffuse"].pixels[..., :3] * environment
            else:
                diffuse_sum = diffuse_sum * p["diffuse"]
            if "environment_specular" in layers:
                specular_total = specular_total + layers["environment_specular"].pixels[..., :3] * environment
            if "reflections" in layers:
                specular_total = specular_total + layers["reflections"].pixels[..., :3] * p.get("reflections", 1.0)
            relit_rgb = (layers["albedo"].pixels[..., :3] * (ambient + diffuse_sum)
                         + specular_total * p["specular"])
            if "indirect" in layers:
                # One diffuse bounce from the splat bundle (already times albedo); `Indirect` rebalances it.
                relit_rgb = relit_rgb + layers["indirect"].pixels[..., :3] * p.get("indirect", 1.0)
            result_rgb = p["mix"] * relit_rgb + (1 - p["mix"]) * bundle_raster.pixels[..., :3]
            return bundle_raster.with_pixels(np.concatenate(
                (result_rgb, bundle_raster.pixels[..., 3:4]), axis=-1))
        if kind == "LightMixer":
            source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            return Evaluator._light_mixer(source, mask, p)
        if kind == "ShuffleCopy":
            first, second = inputs[0], inputs[1]
            if first.display != second.display:
                raise ValueError("ShuffleCopy inputs must have matching formats in M0")
            first = Evaluator._layer_of(kind, first, p.get("layer1", ""))
            second = Evaluator._layer_of(kind, second, p.get("layer2", ""))
            sources = {"in1": first, "in2": second}
            def route(output):
                channels = []
                for channel in "rgba":
                    source_name = p[f"out{output}_{channel}"]
                    source, component = source_name.split(".")
                    channels.append(sources[source].pixels[..., "rgba".index(component):"rgba".index(component)+1])
                return np.concatenate(channels, axis=2).astype(np.float32)
            out1, out2 = route(1), route(2)
            return Raster(out1, first.data, first.display, {"out2": Raster(out2, first.data, first.display)})
        if kind == "ChannelShuffle":
            a, b = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if b is not None and b.display != a.display:
                raise ValueError(
                    f"ChannelShuffle B display window {b.display} does not match A {a.display}; "
                    "no silent resampling is performed")
            # Pointwise over A's rectangle. B is fitted into it rather than unioned: routing a
            # channel out of B cannot enlarge the picture, it can only recolour the picture A
            # already defines. Area of B outside A's data window reads as transparent black,
            # which is what `fit` produces.
            out = a.data
            return Raster(Evaluator._kernel(kind, p, [a.pixels, None if b is None else b.fit(out)],
                                            frame), out, a.display)
        if kind == "LightWrap":
            fg, bg = inputs[0], inputs[1]
            if fg.display != bg.display:
                raise ValueError("LightWrap inputs must have matching formats in M0")
            out = fg.data
            mask = inputs[2] if len(inputs) > 2 else None
            if mask is not None and mask.display != fg.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match LightWrap fg {fg.display}; "
                    "no silent resampling is performed")
            layers = [fg.pixels, bg.fit(out)] + ([] if mask is None else [mask.fit(out)])
            return Raster(Evaluator._kernel(kind, p, layers, frame), out, fg.display)
        if kind == "IBKGizmo":
            fg, plate, bg = inputs[0], inputs[1], (inputs[2] if len(inputs) > 2 else None)
            mask = inputs[3] if len(inputs) > 3 else None
            for other in (plate, bg, mask):
                if other is not None and other.display != fg.display:
                    raise ValueError(
                        f"IBKGizmo inputs must have matching formats in M0 (display {other.display} vs fg {fg.display})")
            out = fg.data
            layers = [fg.pixels, plate.fit(out), None if bg is None else bg.fit(out),
                      None if mask is None else mask.fit(out)]
            return Raster(Evaluator._kernel(kind, p, layers, frame), out, fg.display)
        if kind == "Blend":
            wired = [r for r in inputs[:16] if r is not None]
            if len(wired) < 1:
                raise ValueError("Blend: connect at least one input")
            display = wired[0].display
            if any(r.display != display for r in wired):
                raise ValueError("Blend inputs must have matching formats in M0")
            out = wired[0].data
            for r in wired[1:]:
                out = out.union(r.data)
            mask = inputs[16] if len(inputs) > 16 else None
            if mask is not None and mask.display != display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match Blend {display}; "
                    "no silent resampling is performed")
            layers = [r.fit(out) if r is not None else None for r in inputs[:16]]
            layers += [None] * (16 - len(layers)) + [mask.fit(out) if mask is not None else None]
            return Raster(Evaluator._kernel(kind, p, layers, frame), out, display)
        if kind == "CopyBBox":
            a, b = inputs[0], inputs[1]
            if a.display != b.display:
                raise ValueError("CopyBBox inputs must have matching display windows")
            return Raster(a.fit(b.data), b.data, a.display, a.layers, a.meta)
        if kind in MERGE_LIKE_KINDS and kind != "ZMerge":
            a, b = inputs[0], inputs[1]
            if a.display != b.display:
                # Differing *display* windows is still a format mistake and still raises, exactly
                # as it did before bounding boxes existed. Differing *data* windows is now normal:
                # that is a plate with overscan merged over one without, which must work.
                raise ValueError(f"{kind} inputs must have matching formats in M0")
            out = a.data.union(b.data)
            mask = inputs[2] if len(inputs) > 2 else None
            if mask is not None and mask.display != b.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match {kind} B {b.display}; "
                    "no silent resampling is performed")
            layers = [a.fit(out), b.fit(out)] + ([] if mask is None else [mask.fit(out)])
            return Raster(Evaluator._kernel(kind, p, layers, frame, origin=(out.x, out.y)), out, b.display)
        if kind == "Switch":
            chosen = Evaluator._kernel("Switch", p, [r.pixels if r is not None else None
                                                     for r in inputs[:2]], frame)
            return inputs[int(p["which"])].with_pixels(chosen)
        if kind == "Reformat":
            # Unlike every other MASK_MIX_KINDS member, Reformat's own output *display* window is
            # not `source.display` -- it is the chosen format (docs/PARITY_2D.md) -- so it cannot
            # go through the generic branch below, which always keeps the source's display. mask/
            # mix still follow the same recipe, just gated against the new format's own rectangle.
            source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if mask is not None and mask.display != source.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match source {source.display}; "
                    "no silent resampling is performed")
            reformat_type = p["reformat_type"]
            if reformat_type == "scale":
                factor = float(p["scale"])
                target_w = max(1, int(round(source.display.width * factor)))
                target_h = max(1, int(round(source.display.height * factor)))
            else:
                target_w, target_h = int(p["width"]), int(p["height"])
            target_display = Region(0, 0, target_w, target_h)
            own_box = target_display
            if p["preserve_bbox"]:
                own_box = target_display.union(Evaluator._reformat_transformed_window(
                    source.data, source.display.width, source.display.height,
                    target_w, target_h, p))
            mix = p["mix"]
            gated = mask is not None or mix != 1.0
            out = own_box.union(source.data) if gated else own_box
            filtered = Evaluator._reformat(source.pixels, source.display.width, source.display.height,
                                           target_w, target_h, p, src_box=source.data, dst_box=out)
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), mix)
            return Raster(pixels, out, target_display)
        if kind == "TVIScale":
            # A power-of-two special case of Reformat's own "distort"-fit placement math: since
            # both axes scale by the identical `2**power` factor, "fit"/"fill"/"distort" all agree
            # (the target aspect never drifts from the source's), so `distort` is used directly
            # rather than inventing a separate resize-type knob this legacy node never had.
            source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            if mask is not None and mask.display != source.display:
                raise ValueError(
                    f"Mask display window {mask.display} does not match source {source.display}; "
                    "no silent resampling is performed")
            factor = 2.0 ** int(p["power"])
            target_w = max(1, int(round(source.display.width * factor)))
            target_h = max(1, int(round(source.display.height * factor)))
            target_display = Region(0, 0, target_w, target_h)
            mix = p["mix"]
            gated = mask is not None or mix != 1.0
            out = target_display.union(source.data) if gated else target_display
            reformat_params = {"resize_type": "distort", "center": 1, "flip": 0, "flop": 0,
                               "turn": 0, "filter": p["filter"]}
            filtered = Evaluator._reformat(source.pixels, source.display.width, source.display.height,
                                           target_w, target_h, reformat_params,
                                           src_box=source.data, dst_box=out)
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), mix)
            return Raster(pixels, out, target_display)
        if kind == "Precomp":
            from .core import load_document
            path = p.get("file", "")
            if not path:
                raise ValueError("Precomp: choose a referenced document")
            try:
                ref_doc = load_document(path)
            except OSError:
                raise ValueError(f"Precomp: file not found: {path}") from None
            except ValueError as error:
                raise ValueError(f"Precomp: {path}: {error}") from None
            output_id = p.get("output_node") or ref_doc.get("view")
            if not output_id or output_id not in ref_doc["nodes"]:
                raise ValueError(f"Precomp: no output node {output_id!r} in {path}")
            ref_frame = (frame or 1) + int(p.get("frame_offset", 0))
            # A fresh, disk-cache-free Evaluator: Precomp is a low-priority script-management node
            # (docs/PARITY_2D.md), so correctness on every call outweighs sharing a cache with the
            # outer document, which a second document's node ids could collide with anyway.
            nested = Evaluator(disk=cachetier.DiskCache(enabled=False))
            return nested.evaluate_raster(ref_doc, output_id, frame=ref_frame, tier=1)
        if kind in METADATA_KINDS:
            return Evaluator._metadata_node(kind, p, inputs, frame)
        if kind == "BurnIn":
            return Evaluator._burn_in(p, inputs[0], frame)
        if kind in WINDOW_KINDS:
            return Evaluator._window_node(kind, p, inputs[0])
        if kind == "Cryptomatte":
            return Evaluator._cryptomatte(p, inputs)
        if kind == "Encryptomatte":
            return Evaluator._encryptomatte(p, inputs)
        if kind == "ZMerge":
            a, b = inputs[0], inputs[1]
            if a.display != b.display:
                raise ValueError("ZMerge inputs must have matching formats")
            out = a.data.union(b.data)
            ap, bp = a.fit(out), b.fit(out)
            layer = p.get("depth_layer", "depth.Z")
            az = Evaluator._layer_of(kind, a, layer).fit(out)[..., 0]
            bz = Evaluator._layer_of(kind, b, layer).fit(out)[..., 0]
            if p.get("depth_math") == "1/depth":
                az = 1.0 / np.maximum(az, 1e-6); bz = 1.0 / np.maximum(bz, 1e-6)
            delta = az - bz
            if p.get("depth_math") == "1/depth":
                delta = -delta  # inverse depth grows toward the camera
            smooth = max(0.0, float(p.get("smoothing", 0.0)))
            weight = (delta < 0).astype(np.float32) if smooth == 0 else np.clip(0.5 - delta / (2 * smooth), 0, 1)
            filtered = ap * weight[..., None] + bp * (1 - weight[..., None])
            mask = inputs[2] if len(inputs) > 2 else None
            pixels = Evaluator._apply_mask_mix(bp, filtered, None if mask is None else mask.fit(out), p.get("mix", 1.0))
            # Preserve the source depth units in the exported layer even when comparison used 1/z.
            nearest = np.minimum(Evaluator._layer_of(kind, a, layer).fit(out)[..., 0],
                                 Evaluator._layer_of(kind, b, layer).fit(out)[..., 0])
            depth_pixels = np.repeat(nearest[..., None], 4, axis=2); depth_pixels[..., 3] = 1.0
            layer_name = layer.rsplit(".", 1)[0] if "." in layer else layer
            layers = dict(b.layers or {}); layers.update(a.layers or {})
            layers[layer_name] = Raster(depth_pixels, out, a.display)
            return Raster(pixels, out, a.display, layers, b.meta or a.meta)
        if kind == "ZSlice":
            source = inputs[0]; depth_input = inputs[1] if len(inputs) > 1 else None
            mask = inputs[2] if len(inputs) > 2 else None
            depth_raster = depth_input or Evaluator._layer_of(kind, source, p.get("depth_layer", "depth.Z"))
            depth = depth_raster.fit(source.data)[..., 0]
            near, far = sorted((float(p.get("near", 0)), float(p.get("far", 1))))
            falloff = max(0.0, float(p.get("falloff", 0)))
            if p.get("depth_math") == "1/depth": depth = 1.0 / np.maximum(depth, 1e-6)
            matte = np.ones_like(depth, dtype=np.float32) if falloff == 0 else np.clip((depth - (near-falloff))/falloff, 0, 1) * np.clip(((far+falloff)-depth)/falloff, 0, 1)
            matte = ((depth >= near) & (depth <= far)).astype(np.float32) if falloff == 0 else matte
            if p.get("zslice_output", "matte") == "image": filtered = source.pixels * matte[..., None]
            else: filtered = np.repeat(matte[..., None], 4, axis=2); filtered[..., 3] = matte
            pixels = Evaluator._apply_mask_mix(source.pixels, filtered, None if mask is None else mask.fit(source.data), p.get("mix", 1.0))
            return Raster(pixels, source.data, source.display, source.layers, source.meta)
        if kind == "Remove":
            source = inputs[0]
            if not source.layers: return source
            names = {n.strip() for n in str(p.get("layers", "")).split(",") if n.strip()}
            keep_mode = p.get("remove_operation", "remove") == "keep"
            layers = {name: value for name, value in source.layers.items() if ((name in names) == keep_mode)}
            return Raster(source.pixels, source.data, source.display, layers, source.meta)
        if kind == "ZDefocus":
            source = inputs[0]
            depth_input = inputs[1] if len(inputs) > 1 else None
            kernel_input = inputs[2] if len(inputs) > 2 else None
            mask = inputs[3] if len(inputs) > 3 else None
            depth_raster = depth_input or Evaluator._layer_of(kind, source, p.get("depth_layer", "depth.Z"))
            depth = depth_raster.fit(source.data)[..., 0]
            filtered = Evaluator._zdefocus(source.pixels, depth, p, None if kernel_input is None else kernel_input.pixels)
            pixels = Evaluator._apply_mask_mix(source.pixels, filtered,
                                               None if mask is None else mask.fit(source.data), p.get("mix", 1.0))
            return Raster(pixels, source.data, source.display, source.layers, source.meta)
        if kind in UV_KINDS:
            return Evaluator._uv_node(kind, p, inputs)
        if kind in ("SplineWarp", "GridWarp", "GridWarpTracker"):
            return Evaluator._warp_node(kind, p, inputs, data, frame)
        if kind == "LevelSet":
            return Evaluator._levelset(inputs[0], p)
        if kind == "Convolve":
            source, kernel = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if kernel is None: raise ValueError("Convolve requires a kernel image")
            if mask is not None and mask.display != source.display:
                raise ValueError(f"Mask display window {mask.display} does not match source {source.display}")
            out = source.data
            filtered = Evaluator._convolve(source.fit(out), kernel.pixels, p)
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), p.get("mix", 1.0))
            return Raster(pixels, out, source.display)
        if kind == "Expression":
            source, second = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            mask = inputs[2] if len(inputs) > 2 else None
            if second is not None and second.display != source.display:
                raise ValueError("Expression second input must match the primary image format")
            if mask is not None and mask.display != source.display:
                raise ValueError("Expression mask must match the primary image format")
            out = source.data
            filtered = Evaluator._expression(source.fit(out), None if second is None else second.fit(out),
                                              p, frame, (out.x, out.y))
            pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                               None if mask is None else mask.fit(out), p.get("mix", 1.0))
            return Raster(pixels, out, source.display, source.layers, source.meta)
        if kind == "MatchGrade":
            source, target = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            baked = bool(p.get("match_analyzed", 0))
            if target is None and not baked:
                raise ValueError("MatchGrade needs a connected reference, or Analyze once to bake gain and offset")
            out = source.data
            if target is not None and source.display != target.display:
                raise ValueError("MatchGrade source and target formats must match")
            target_fit = target.fit(out) if target is not None else None
            pixels = Evaluator._match_grade(source.fit(out), target_fit, p)
            pixels = Evaluator._apply_mask_mix(source.fit(out), pixels,
                                               None if mask is None else mask.fit(out), p.get("mix", 1.0))
            return Raster(pixels, out, source.display)
        if kind == "RotoPaint":
            from . import paint
            source = inputs[0]
            reveal = inputs[1] if len(inputs) > 1 else None
            if reveal is not None and reveal.display != source.display:
                raise ValueError("RotoPaint input2 must match the plate format")
            out = source.data
            mask = inputs[2] if len(inputs) > 2 else None
            if mask is not None and mask.display != source.display:
                raise ValueError("RotoPaint mask must match the plate format")
            painted = paint.rasterise(source.fit(out), data or [], frame,
                                      source=source.fit(out), reveal=None if reveal is None else reveal.fit(out))
            pixels = Evaluator._apply_mask_mix(source.fit(out), painted,
                                               None if mask is None else mask.fit(out), p.get("mix", 1.0))
            return Raster(pixels, out, source.display, source.layers, source.meta)
        if kind in ("Flare", "Glint", "Sparkles", "GodRays", "VolumeRays", "ScannedGrain"):
            source = inputs[0]
            plate = inputs[1] if kind == "ScannedGrain" and len(inputs) > 1 else None
            matte = inputs[1] if kind == "VolumeRays" and len(inputs) > 1 else None
            mask_index = 2 if kind in ("ScannedGrain", "VolumeRays") else 1
            mask = inputs[mask_index] if len(inputs) > mask_index else None
            if plate is not None and plate.display != source.display:
                raise ValueError("ScannedGrain plate must match source format")
            out = source.data
            base = source.fit(out)
            if kind == "Flare": filtered = Evaluator._flare(base, p, (out.x, out.y),
                                                               (source.display.width, source.display.height))
            elif kind == "Glint": filtered = Evaluator._glint(base, p)
            elif kind == "Sparkles": filtered = Evaluator._sparkles(base, p, frame, (out.x, out.y))
            elif kind in ("GodRays", "VolumeRays"):
                filtered = Evaluator._godrays(base, p, (out.x, out.y), None if matte is None else matte.fit(out))
            else: filtered = Evaluator._scanned_grain(base, None if plate is None else plate.fit(out), p, frame)
            pixels = Evaluator._apply_mask_mix(base, filtered, None if mask is None else mask.fit(out), p.get("mix", 1.0))
            return Raster(pixels, out, source.display, source.layers, source.meta)
        if kind in ("SplineWarp", "GridWarp", "GridWarpTracker"):
            return Evaluator._warp_node(kind, p, inputs, data, frame)
        if kind == "LevelSet":
            return Evaluator._levelset(inputs[0], p)
        if kind == "ScreenKeyer":
            # Image plus the optional inside, outside, clean-plate and mix-mask inputs, all fitted to the
            # image's own rectangle (pointwise after the matte's shrink and softness, so no window grows).
            source = inputs[0]
            others = [inputs[i] if len(inputs) > i else None for i in (1, 2, 3, 4)]
            for name, other in zip(("inside", "outside", "clean", "mask"), others):
                if other is not None and other.display != source.display:
                    raise ValueError(
                        f"ScreenKeyer {name} display window {other.display} does not match the image "
                        f"{source.display}; no silent resampling is performed")
            out = source.data
            layers = [source.pixels] + [None if other is None else other.fit(out) for other in others]
            return Raster(Evaluator._kernel(kind, p, layers, frame), out, source.display)
        if kind in MASK_MIX_KINDS:
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
            if (cancel is not None and kind in _ROW_CHUNKED_MASK_MIX_KINDS
                    and out.height > _CANCEL_CHUNK_ROWS):
                pixels = Evaluator._run_row_chunked(kind, p, source, mask, mix, out, frame, cancel, progress)
            else:
                filtered = Evaluator._filtered_pixels(kind, p, source, out, frame)
                pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                                   None if mask is None else mask.fit(out), mix)
            return Raster(pixels, out, source.display)
        # Pointwise and pass-through kinds: Viewer, Write, NoOp, Dot, Shuffle, Premult, Unpremult.
        source = inputs[0]
        if kind in ("MinColor", "Sampler", "CurveTool"):
            return source
        if kind in ("Viewer", "Write", "NoOp", "PostageStamp", "Output", "Profile"):
            return source   # taps hand the raster on whole, so named layers reach a downstream Write
        if kind == "Shuffle" and p.get("layer"):
            layer = Evaluator._layer_of(kind, source, p["layer"])
            return Raster(Evaluator._kernel(kind, p, [layer.pixels], frame), layer.data, source.display)
        return source.with_pixels(Evaluator._kernel(kind, p, [source.pixels], frame))

    @staticmethod
    def _metadata_node(kind, p, inputs, frame):
        """ViewMetaData, ModifyMetaData, CopyMetaData, CompareMetaData and AddTimeCode: the pixels, windows
        and layers of the first input untouched, with its metadata edited (docs/PARITY_2D.md)."""
        source, other = inputs[0], (inputs[1] if len(inputs) > 1 else None)
        meta = source.meta
        if kind == "ModifyMetaData":
            meta = metadata.modify(meta, p["edits"], frame)
        elif kind == "CopyMetaData" and other is not None:
            meta = metadata.copy(meta, other.meta, p["keys"])
        elif kind == "AddTimeCode":
            meta = dict(meta or {})
            meta[metadata.TIMECODE_KEY] = metadata.timecode_at(
                p["timecode"], p["fps"], p["start_frame"], frame, bool(p["drop_frame"]))
        if meta == source.meta:
            return source
        return Raster(source.pixels, source.data, source.display, source.layers, meta or {})

    @staticmethod
    def _burn_in(p, source, frame):
        """Text overlay on the image: five slots (corners and centre) of literal text with `[frame]` and
        `[metadata key]` substitutions, drawn by the Text node's own rasteriser, over optional top and
        bottom bars."""
        display, box = source.display, source.data
        width, height = display.width, display.height
        size = float(p["font_size"])
        margin, band = float(p["margin"]), max(1, int(round(size * 2.0)))
        slots = (("top_left", "left", 0.0, band, True), ("top_right", "right", 0.0, band, True),
                 ("bottom_left", "left", height - band, band, True),
                 ("bottom_right", "right", height - band, band, True), ("center", "center", 0.0, height, False))
        overlay = np.zeros((box.height, box.width, 4), np.float32)
        texts = [(name, justify, top, extent, bar, metadata.expand(str(p[name]), source.meta, frame))
                 for name, justify, top, extent, bar in slots]
        if int(p["bar"]):
            opacity = np.float32(np.clip(p["bar_opacity"], 0.0, 1.0))
            filled = {name: bool(text) for name, _, _, _, _, text in texts}
            for top, wanted in ((0, filled["top_left"] or filled["top_right"]),
                                (height - band, filled["bottom_left"] or filled["bottom_right"])):
                rows = Region(0, top, width, band).intersect(box)
                if wanted and not rows.is_empty:
                    overlay[rows.y - box.y:rows.bottom - box.y, rows.x - box.x:rows.right - box.x, 3] = opacity
        for name, justify, top, extent, bar, text in texts:
            if not text:
                continue
            text_params = {"message": text, "font": p["font"], "font_size": size, "justify": justify,
                           "box_x": margin, "box_y": top, "box_width": max(1.0, width - 2 * margin),
                           "box_height": float(extent), "red": p["red"], "green": p["green"],
                           "blue": p["blue"], "alpha": p["alpha"]}
            shape = Evaluator._text_shape(text_params, box.x, box.y, box.width, box.height)
            overlay = Evaluator._composite_shape_over(shape, overlay)
        pixels = Evaluator._composite_shape_over(overlay, source.pixels)
        return Raster(pixels, source.data, source.display, source.layers, source.meta)

    @staticmethod
    def _layer_of(kind, raster, name):
        """The named layer of `raster` as a Raster, or the raster itself for "" / "rgba"."""
        if not name or name == "rgba":
            return raster
        layers = raster.layers or {}
        if name in layers:
            return layers[name]
        # Nuke-style component paths (for example depth.Z) select a channel from a named
        # multichannel layer; the layer itself remains available by its bare name.
        if "." in name:
            layer_name, component = name.rsplit(".", 1)
            if layer_name in layers and component.upper() in ("R", "G", "B", "A", "X", "Y", "Z", "W"):
                layer = layers[layer_name]
                index = {"R": 0, "X": 0, "G": 1, "Y": 1, "B": 2, "Z": 2, "A": 3, "W": 3}[component.upper()]
                if index < layer.pixels.shape[2]:
                    component_pixels = np.repeat(layer.pixels[..., index:index+1], 4, axis=2)
                    component_pixels[..., 3] = 1.0
                    return Raster(component_pixels, layer.data, layer.display)
        have = ", ".join(["rgba", *layers]) if layers else "none (wire a multichannel Read or Render3D)"
        raise ValueError(f"{kind}: no layer {name!r} on the input; available: {have}")

    @staticmethod
    def _cryptomatte(p, inputs):
        """Cryptomatte: the matte of the listed ids from the input's Cryptomatte layer set.

        The matte is the coverage summed over every rank whose id is listed (`nodebased/cryptomatte.py`).
        View `final` keeps the input's colour and puts the matte in alpha (straight, not premultiplied),
        `matte` writes the matte as grey with the same alpha, `colors` paints every id its own colour.
        An empty matte list is an empty matte. The layers ride on the whole-image raster only, so the
        node sits on a Read (or a tap of one) and never runs on the tile path.
        """
        from . import cryptomatte
        source, mask = inputs[0], (inputs[1] if len(inputs) > 1 else None)
        if mask is not None and mask.display != source.display:
            raise ValueError(f"Mask display window {mask.display} does not match source {source.display}; "
                             "no silent resampling is performed")
        name, members = cryptomatte.choose_set(source, p.get("crypto_layer", ""))
        for layer in members:
            if source.layers[layer].data != source.data:
                raise ValueError(f"Cryptomatte: layer {layer!r} does not cover the same window as the image")
        ids = cryptomatte.parse_matte_list(p.get("matte_list", ""), cryptomatte.manifest_for(source, name))
        view = p.get("crypto_view", "final")
        matte = cryptomatte.matte(source, members, ids)
        result = source.pixels.copy()
        if view == "matte":
            result[..., :3] = matte[..., None]
            result[..., 3] = matte
        elif view == "colors":
            result[..., :3] = cryptomatte.preview(source, members)
            result[..., 3] = 1.0
        else:
            result[..., 3] = matte
        pixels = Evaluator._apply_mask_mix(source.pixels, result, None if mask is None else mask.fit(source.data),
                                           p.get("mix", 1.0))
        return Raster(pixels, source.data, source.display)

    @staticmethod
    def _encryptomatte(p, inputs):
        """Encryptomatte: ranked Cryptomatte layers written from up to eight named mattes, in the
        exact channel layout `nodebased/cryptomatte.py` and the `Cryptomatte` node read (id in R/B,
        coverage in G/A of each `<layer_name>NN` group, best coverage first) plus the matching
        header manifest (`nodebased/cryptomatte3d.cryptomatte_header`, shared with Render3D's
        writer side). `matte{i}`'s alpha channel is that object's coverage; `id{i}` names it. An
        unwired matte, or a wired one with no name, contributes nothing. The image passes through
        untouched: only its layers and header metadata change, so a plain `image` slot bypass
        already does the right thing.
        """
        from . import cryptomatte, cryptomatte3d
        source = inputs[0]
        layer_name = str(p.get("layer_name") or "crypto_object")
        entries = []
        for i, matte in enumerate(inputs[1:9]):
            name = str(p.get(f"id{i}") or "").strip()
            if matte is None or not name:
                continue
            if matte.display != source.display:
                raise ValueError(f"Encryptomatte: matte {i} display window {matte.display} does not "
                                 f"match source {source.display}; no silent resampling is performed")
            entries.append((name, matte.fit(source.data)[..., 3]))
        if not entries:
            return Raster(source.pixels, source.data, source.display, source.layers, source.meta)
        ids = np.array([cryptomatte.name_to_bits(name) for name, _ in entries], np.uint32)
        coverage = np.stack([cov for _, cov in entries], axis=-1).astype(np.float32)   # (H, W, N)
        order = np.argsort(-coverage, axis=-1, kind="stable")
        ranked_ids = ids[order]
        ranked_cov = np.take_along_axis(coverage, order, axis=-1)
        levels = max(2, -(-len(entries) // 2) * 2)   # even, at least 2: one group per two ranks
        pad = levels - len(entries)
        if pad:
            ranked_ids = np.pad(ranked_ids, ((0, 0), (0, 0), (0, pad)))
            ranked_cov = np.pad(ranked_cov, ((0, 0), (0, 0), (0, pad)))
        new_layers = {}
        for group in range(levels // 2):
            layer = np.zeros((*ranked_cov.shape[:2], 4), np.float32)
            layer[..., 0] = np.ascontiguousarray(ranked_ids[..., 2 * group]).view(np.float32)
            layer[..., 1] = ranked_cov[..., 2 * group]
            layer[..., 2] = np.ascontiguousarray(ranked_ids[..., 2 * group + 1]).view(np.float32)
            layer[..., 3] = ranked_cov[..., 2 * group + 1]
            new_layers[f"{layer_name}{group:02d}"] = Raster(layer, source.data, source.display)
        manifest = {name: format(int(cryptomatte.name_to_bits(name)), "08x") for name, _ in entries}
        header = cryptomatte3d.cryptomatte_header(
            {layer_name: {"key": cryptomatte.set_key(layer_name), "manifest": manifest}})
        merged_layers = {**(source.layers or {}), **new_layers}
        merged_meta = {**(source.meta or {}), **header}
        return Raster(source.pixels, source.data, source.display, merged_layers, merged_meta)

    _UV_SLOT = {"R": 0, "G": 1, "B": 2, "A": 3}

    @staticmethod
    def _uv_node(kind, p, inputs):
        """STMap, IDistort and VectorBlur: image + optional uv raster + optional mask, whole image.

        The map is `uv_layer` of the uv input (or of the image input when uv is not wired); an empty
        `uv_layer` takes the uv input's own channels. Windows follow the Blur/Transform precedents:
        STMap's output covers the map's data window (Nuke's rule: it exists where the map does),
        IDistort and VectorBlur keep the source's data window. A gated node (mask or mix < 1) blends
        against the untouched source, so the source rectangle joins the output rectangle.
        """
        source = inputs[0]
        uv_in = inputs[1] if len(inputs) > 1 else None
        mask = inputs[2] if len(inputs) > 2 else None
        if uv_in is None and not p.get("uv_layer"):
            raise ValueError(f"{kind}: connect the uv input or choose a uv_layer")
        uv = Evaluator._layer_of(kind, uv_in if uv_in is not None else source, p.get("uv_layer", ""))
        if mask is not None and mask.display != source.display:
            raise ValueError(f"Mask display window {mask.display} does not match source {source.display}; "
                             "no silent resampling is performed")
        mix = p.get("mix", 1.0)
        base = uv.data if kind == "STMap" else source.data
        out = base.union(source.data) if (mask is not None or mix != 1.0) else base
        if out.is_empty:
            return Raster(np.zeros((out.height, out.width, 4), np.float32), out, source.display)
        u_index, v_index = Evaluator._UV_SLOT[p["u_channel"]], Evaluator._UV_SLOT[p["v_channel"]]
        uv_pixels = uv.fit(out)
        u = np.nan_to_num(uv_pixels[..., u_index], nan=0.0, posinf=0.0, neginf=0.0)
        v = np.nan_to_num(uv_pixels[..., v_index], nan=0.0, posinf=0.0, neginf=0.0)
        if kind == "STMap":
            filtered = Evaluator._stmap(source, u, v, p, out)
        elif kind == "IDistort":
            filtered = Evaluator._idistort(source, u, v, p, out)
        else:
            filtered = Evaluator._vector_blur(source.fit(out), u, v, p)
        if uv.data != out:
            # Outside the map's own window there is no map, so there is nothing to warp or blur.
            inside = np.zeros((out.height, out.width, 1), np.float32)
            overlap = out.intersect(uv.data)
            if not overlap.is_empty:
                inside[overlap.y - out.y:overlap.bottom - out.y, overlap.x - out.x:overlap.right - out.x] = 1.0
            filtered = filtered * inside
        pixels = Evaluator._apply_mask_mix(source.fit(out), filtered,
                                           None if mask is None else mask.fit(out), mix)
        return Raster(pixels, out, source.display)

    @staticmethod
    def _gridwarp_history_raster(vectors, frame):
        if vectors is None:
            return None
        prefix = f"gridwarp.history.{int(frame)}|"
        layers = {name[len(prefix):]: layer for name, layer in (vectors.layers or {}).items()
                  if name.startswith(prefix)}
        return Raster(vectors.pixels, vectors.data, vectors.display, layers, vectors.meta) if layers else None

    @staticmethod
    def _gridwarp_vector_valid(point, vectors, p, frame):
        if vectors is None:
            return True
        from .opticalflow import _sample
        layers = vectors.layers or {}
        forward_name = p.get("forward_layer", "smartvector.forward")
        backward_name = p.get("backward_layer", "smartvector.backward")
        forward = layers.get(forward_name) or layers.get("vector.forward") or layers.get("smartvector.forward")
        backward = layers.get(backward_name) or layers.get("vector.backward") or layers.get("smartvector.backward")
        selected = forward if frame >= float(p.get("reference_frame", 1)) else backward
        xy = np.asarray(point, dtype=np.float32).reshape(1, 1, 2)
        occ = layers.get("vector.occlusion") or layers.get("smartvector.occlusion")
        if occ is not None and float(_sample(occ.pixels[..., :1], xy[..., 0], xy[..., 1])[0, 0, 0]) >= .5:
            return False
        if selected is None:
            return True
        if selected.pixels.shape[-1] >= 3 and float(_sample(selected.pixels[..., 2:3], xy[..., 0], xy[..., 1])[0, 0, 0]) >= .5:
            return False
        if forward is not None and backward is not None:
            f = _sample(forward.pixels[..., :2], xy[..., 0], xy[..., 1])
            end = xy + f
            b = _sample(backward.pixels[..., :2], end[..., 0], end[..., 1])
            if float(np.linalg.norm(f + b)) > float(p.get("fb_threshold", 1.0)):
                return False
        return True

    @staticmethod
    def _gridwarp_fit_tracker(samples):
        src, dst = (np.asarray(x, dtype=np.float64) for x in samples)
        if len(src) >= 3 and np.linalg.matrix_rank(np.column_stack((src, np.ones(len(src))))) >= 3:
            coeff, _, _, _ = np.linalg.lstsq(np.column_stack((src, np.ones(len(src)))), dst, rcond=None)
            matrix = np.array([[coeff[0, 0], coeff[1, 0]], [coeff[0, 1], coeff[1, 1]]])
            offset = coeff[2]
        elif len(src) >= 2:
            a, b = src.mean(axis=0), dst.mean(axis=0)
            x, y = src - a, dst - b
            den = float((x*x).sum())
            cosine = float((x*y).sum()) / den if den > 1e-12 else 1.0
            sine = float((x[:, 0]*y[:, 1] - x[:, 1]*y[:, 0]).sum()) / den if den > 1e-12 else 0.0
            matrix = np.array([[cosine, -sine], [sine, cosine]])
            offset = b - matrix @ a
        else:
            matrix = np.eye(2)
            offset = dst[0] - src[0] if len(src) else np.zeros(2)
        return matrix, offset

    @staticmethod
    def _gridwarp_tracker_controls(source, p, data, vectors=None, frame=1):
        """Resolve tracker/vector motion and hold each control at its last valid sample."""
        from .warps import default_grid
        src_grid = default_grid(source.display.width, source.display.height, p["rows"], p["columns"])
        grids = data
        if grids and "tracker_points" in grids:
            sample_src, current_dst = (np.asarray(x, dtype=float) for x in grids["tracker_points"])
            indices = grids.get("tracker_indices", list(range(len(sample_src))))
            history = grids.get("tracker_history", [])
            chosen = current_dst.copy()
            resolved = np.zeros(len(sample_src), dtype=bool)
            for history_frame, positions in history:
                history_frame = int(history_frame)
                candidate_vectors = vectors if history_frame == int(frame) else Evaluator._gridwarp_history_raster(vectors, history_frame)
                for k, track_index in enumerate(indices):
                    if resolved[k] or k >= len(positions) or positions[k] is None:
                        continue
                    point = np.asarray(positions[k], dtype=float)
                    if Evaluator._gridwarp_vector_valid(sample_src[k], candidate_vectors, p, history_frame):
                        chosen[k] = point
                        resolved[k] = True
            chosen[~resolved] = sample_src[~resolved]
            matrix, offset = Evaluator._gridwarp_fit_tracker((sample_src, chosen))
            grids = {"tracker_affine": (matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1],
                                       offset[0], offset[1]),
                     "tracker_samples": (sample_src.tolist(), chosen.tolist())}
        if grids and "tracker_affine" in grids:
            a, b, c, d, tx, ty = grids["tracker_affine"]
            matrix = np.array([[a, b], [c, d]], dtype=np.float64)
            src_points = np.asarray(src_grid, dtype=float)
            dst_points = np.einsum("ij,...j->...i", matrix, src_points) + (tx, ty)
            if p.get("local_motion", 0.0) > 0 and "tracker_samples" in grids:
                sample_src, sample_dst = (np.asarray(x, dtype=float) for x in grids["tracker_samples"])
                residual = sample_dst - (sample_src @ matrix.T + np.array([tx, ty]))
                distance = np.linalg.norm(src_points.reshape(-1, 2)[:, None, :] - sample_src[None, :, :], axis=-1)
                weights = 1.0 / np.maximum(distance, 1e-3) ** 2
                local = (weights @ residual) / np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
                dst_points += float(p.get("local_motion", 0.0)) * local.reshape(dst_points.shape)
            return {"source": src_grid, "destination": dst_points.tolist()}
        if grids and "tracker_pose" in grids:
            c, s, ox, oy = grids["tracker_pose"]
            matrix = np.array([[c, -s], [s, c]])
            dst_grid = [[(matrix @ np.asarray(point, dtype=float) + (ox, oy)).tolist() for point in row]
                        for row in src_grid]
            return {"source": src_grid, "destination": dst_grid}
        if p.get("drive", "tracker") == "smartvector":
            points = np.asarray(src_grid, dtype=np.float32)
            history_frames = []
            if vectors is not None:
                for name in (vectors.layers or {}):
                    if name.startswith("gridwarp.history."):
                        try:
                            history_frames.append(int(name.split(".")[2].split("|")[0]))
                        except (IndexError, ValueError):
                            pass
            reference = float(p.get("reference_frame", 1))
            candidates = [int(frame)] + sorted(set(history_frames), key=lambda f: abs(f - int(frame)))
            resolved = np.zeros(points.shape[:-1], dtype=bool)
            destination = points.copy()
            from .opticalflow import _sample
            for candidate_frame in candidates:
                candidate_vectors = vectors if candidate_frame == int(frame) else Evaluator._gridwarp_history_raster(vectors, candidate_frame)
                layers = {} if candidate_vectors is None else (candidate_vectors.layers or {})
                layer_name = (p.get("forward_layer", "smartvector.forward") if candidate_frame >= reference
                              else p.get("backward_layer", "smartvector.backward"))
                layer = layers.get(layer_name) or layers.get("vector.forward" if candidate_frame >= reference else "vector.backward")
                if layer is None:
                    continue
                field = layer.pixels[..., :2]
                if candidate_frame >= reference:
                    motion = _sample(field, points[..., 0], points[..., 1])
                    candidate_destination = points + motion
                else:
                    candidate_destination = points.copy()
                    for _ in range(8):
                        backward = _sample(field, candidate_destination[..., 0], candidate_destination[..., 1])
                        candidate_destination = points - backward
                valid = np.ones(points.shape[:-1], dtype=bool)
                occ = layers.get("vector.occlusion") or layers.get("smartvector.occlusion")
                if occ is not None:
                    valid &= _sample(occ.pixels[..., :1], points[..., 0], points[..., 1])[..., 0] < .5
                if layer.pixels.shape[-1] >= 3:
                    valid &= _sample(layer.pixels[..., 2:3], points[..., 0], points[..., 1])[..., 0] < .5
                forward = layers.get(p.get("forward_layer", "smartvector.forward")) or layers.get("vector.forward")
                backward_layer = layers.get(p.get("backward_layer", "smartvector.backward")) or layers.get("vector.backward")
                if forward is not None and backward_layer is not None:
                    motion_f = _sample(forward.pixels[..., :2], points[..., 0], points[..., 1])
                    reverse = _sample(backward_layer.pixels[..., :2],
                                      (points + motion_f)[..., 0], (points + motion_f)[..., 1])
                    valid &= np.linalg.norm(motion_f + reverse, axis=-1) <= float(p.get("fb_threshold", 1.0))
                take = valid & ~resolved
                destination[take] = candidate_destination[take]
                resolved |= valid
            return {"source": src_grid, "destination": destination.tolist()}
        return {"source": src_grid, "destination": src_grid}

    @staticmethod
    def _warp_node(kind, p, inputs, data, frame=1):
        """Evaluate curve/grid controls as an inverse-sampled image or normalized STMap."""
        from . import warps
        source = inputs[0]
        mask = inputs[1] if len(inputs) > 1 else None
        if mask is not None and mask.display != source.display:
            raise ValueError(f"Mask display window {mask.display} does not match source {source.display}")
        if kind in ("GridWarp", "GridWarpTracker"):
            grids = data
            if kind == "GridWarpTracker":
                grids = Evaluator._gridwarp_tracker_controls(source, p, grids,
                                                               inputs[2] if len(inputs) > 2 else None, frame)
            if not grids or not grids.get("source") or not grids.get("destination"):
                default = warps.default_grid(source.display.width, source.display.height,
                                             p["rows"], p["columns"])
                grids = {"source": default, "destination": default}
            src, dst = warps.grid_controls(grids["source"], grids["destination"])
            sigma = max(1.0, min(source.display.width / max(1, p["columns"] - 1),
                                 source.display.height / max(1, p["rows"] - 1)) * 0.55)
        else:
            src_parts, dst_parts = [], []
            for pair in data or []:
                src_parts.append(warps.sample_curve(pair["source"], p["curve_resolution"]))
                dst_parts.append(warps.sample_curve(pair["destination"], p["curve_resolution"]))
            src = np.concatenate(src_parts) if src_parts else np.zeros((0, 2), np.float64)
            dst = np.concatenate(dst_parts) if dst_parts else np.zeros((0, 2), np.float64)
            sigma = max(1.0, min(source.display.width, source.display.height) / 8.0)
            if p.get("root_warp", "A") == "B":
                src, dst = dst, src
        mix = float(p.get("mix", 1.0))
        if len(src):
            control_source, control_destination = src, dst
        else:
            control_source = control_destination = np.zeros((0, 2), np.float64)
        out = source.data
        if p.get("bbox") == "union" and len(control_destination):
            low = np.floor(control_destination.min(axis=0)).astype(int)
            high = np.ceil(control_destination.max(axis=0)).astype(int) + 1
            from .tiers import Region
            out = out.union(Region(int(low[0]), int(low[1]), int(high[0]-low[0]), int(high[1]-low[1])))
        yy, xx = np.mgrid[out.y:out.bottom, out.x:out.right]
        x, y = xx.astype(np.float64) + 0.5, yy.astype(np.float64) + 0.5
        field = (warps.displacement_field(control_source, control_destination, x, y, sigma)
                 if len(control_source) else np.zeros(x.shape + (2,), np.float32))
        sx = x - field[..., 0] - source.data.x - 0.5
        sy = y - field[..., 1] - source.data.y - 0.5
        warped = Evaluator._resample(source.pixels, sx.astype(np.float32), sy.astype(np.float32), p["filter"])
        if kind == "SplineWarp" and p.get("output", "image") == "stmap":
            mapped_x, mapped_y = x - field[..., 0], y - field[..., 1]
            result = np.zeros((out.height, out.width, 4), np.float32)
            result[..., 0] = mapped_x / source.display.width
            result[..., 1] = 1.0 - mapped_y / source.display.height
            result[..., 3] = 1.0
            identity = np.zeros_like(result)
            identity[..., 0] = x / source.display.width
            identity[..., 1] = 1.0 - y / source.display.height
            identity[..., 3] = 1.0
            blended = Evaluator._apply_mask_mix(identity, result,
                                                  None if mask is None else mask.fit(out), mix)
            return Raster(blended, out, source.display)
        if mix == 0.0:
            return source
        pixels = Evaluator._apply_mask_mix(source.fit(out), warped,
                                           None if mask is None else mask.fit(out), mix)
        return Raster(pixels, out, source.display, source.layers, source.meta)

    @staticmethod
    def _stmap(source, u, v, p, out):
        """Absolute remap: output pixel gets the source sampled at (u * width, (1 - v) * height).

        u and v are normalised to the source's display window, v runs bottom to top (Nuke's
        convention), and pixel centres sit at half integers, so a map holding each pixel's own
        centre is the identity.
        """
        width, height = source.display.width, source.display.height
        sx = u.astype(np.float64) * width - 0.5 - source.data.x
        sy = (1.0 - v.astype(np.float64)) * height - 0.5 - source.data.y
        if p.get("uv_outside", "black") == "clamp":
            sx = np.clip(sx, 0.0, source.data.width - 1.0)
            sy = np.clip(sy, 0.0, source.data.height - 1.0)
        return Evaluator._resample(source.pixels, sx.astype(np.float32), sy.astype(np.float32),
                                   p["filter"]).astype(np.float32)

    @staticmethod
    def _idistort(source, u, v, p, out):
        """Relative pixel offsets: d = (uv + uv_offset) * uv_scale and out(p) = source(p - d), so a
        positive u moves the picture right and a positive v moves it down (image y points down),
        the same way a forward motion vector carries a pixel."""
        dx = (u.astype(np.float64) + p["uv_offset_x"]) * p["uv_scale_x"]
        dy = (v.astype(np.float64) + p["uv_offset_y"]) * p["uv_scale_y"]
        gx = np.arange(out.width, dtype=np.float64)[None, :] + out.x
        gy = np.arange(out.height, dtype=np.float64)[:, None] + out.y
        sx = (gx - dx - source.data.x).astype(np.float32)
        sy = (gy - dy - source.data.y).astype(np.float32)
        return Evaluator._resample(source.pixels, sx, sy, p["filter"]).astype(np.float32)

    @staticmethod
    def _vector_blur(src, u, v, p):
        """Directional blur along a per-pixel motion vector, in pixels per frame.

        The vector is (u, v) * vector_scale, its length capped at max_length (0 = uncapped). Each
        pixel averages bilinear samples of the source at `p - t*vec` for t from vector_offset to
        vector_offset + 1 in steps of about one pixel (`forward`: the streak follows the motion),
        or at `p + t*vec` (`backward`). Destination mode re-reads the field at the estimated
        destination position before choosing the image sample. `vector_alpha` "weighted" averages the straight colour with
        the samples' alpha as weight and keeps the pixel's own alpha (the matte does not smear).
        """
        vx = u.astype(np.float64) * p["vector_scale"]
        vy = v.astype(np.float64) * p["vector_scale"]
        length = np.hypot(vx, vy)
        cap = float(p["max_length"])
        if cap > 0:
            shrink = np.where(length > cap, cap / np.maximum(length, 1e-12), 1.0)
            vx, vy, length = vx * shrink, vy * shrink, np.minimum(length, cap)
        steps = np.maximum(1, np.ceil(length - 1e-9)).astype(np.int32)
        sign = 1.0 if p["vector_method"] == "forward" else -1.0
        height, width = src.shape[:2]
        ix = np.arange(width, dtype=np.float64)[None, :]
        iy = np.arange(height, dtype=np.float64)[:, None]
        weighted = p.get("vector_alpha", "none") == "weighted"
        total = np.zeros(src.shape, np.float32)
        norm = np.zeros(src.shape[:2] + (1,), np.float32)
        fixed_samples = int(p["samples"]) if p.get("samples") else 0
        sample_count = fixed_samples if fixed_samples else int(steps.max()) + 1
        for k in range(sample_count):
            live = np.ones(src.shape[:2], dtype=bool) if fixed_samples else (k <= steps)
            t = (float(p["vector_offset"]) + (0.5 if fixed_samples == 1 else k / (fixed_samples - 1))
                 if fixed_samples else float(p["vector_offset"]) + k / steps)
            if p.get("vector_sampling", "source") == "destination":
                from .opticalflow import _sample
                guess_x = ix - sign * t * vx
                guess_y = iy - sign * t * vy
                sample_vx = _sample(vx, guess_x, guess_y)
                sample_vy = _sample(vy, guess_x, guess_y)
                sx = ix - sign * t * sample_vx
                sy = iy - sign * t * sample_vy
            else:
                sx, sy = ix - sign * t * vx, iy - sign * t * vy
            sample = Evaluator._resample(src, sx.astype(np.float32), sy.astype(np.float32), "bilinear")
            weight = live[..., None].astype(np.float32)
            if weighted:
                weight = weight * sample[..., 3:4]
            total += sample * live[..., None].astype(np.float32)
            norm += weight
        if not weighted:
            return (total / np.maximum(norm, 1e-12)).astype(np.float32)
        result = np.zeros_like(src)
        straight = np.where(norm > 0, total[..., :3] / np.maximum(norm, 1e-12), 0.0)
        result[..., :3] = straight * src[..., 3:4]
        result[..., 3:4] = src[..., 3:4]
        return result

    @staticmethod
    def _motion_layer(source, name, output_data):
        """Read the chosen VectorToMotion layer in pixel units, when present on the beauty."""
        layer = (source.layers or {}).get(name)
        if layer is None:
            return None
        return layer.fit(output_data)[..., :2].astype(np.float32)

    @staticmethod
    def _motion_flow(source, first, second, output_data, params):
        field = Evaluator._motion_layer(source, params.get("vector_layer", "vector.forward"), output_data)
        if field is not None:
            return field
        from .opticalflow import flow_pair
        flow, _, _ = flow_pair(first, second, vector_detail=4, smoothness=1.0,
                               backend=params.get("flow_backend", "auto"))
        return flow

    @staticmethod
    def _camera_depth_flow(depth, camera, camera_start, camera_end, display, data):
        """Project a depth plate through two shutter cameras to get a per-pixel screen-space flow."""
        from . import scene3d
        depth = np.asarray(depth, np.float32)
        if depth.shape != (data.height, data.width):
            raise ValueError("MotionBlur3D: depth dimensions must match the image")
        height, width = display.height, display.width
        yy, xx = np.mgrid[:data.height, :data.width].astype(np.float32)
        canvas_x = xx + np.float32(data.x + 0.5)
        canvas_y = yy + np.float32(data.y + 0.5)
        ndc_x = canvas_x * np.float32(2.0 / max(width, 1)) - 1.0
        ndc_y = 1.0 - canvas_y * np.float32(2.0 / max(height, 1))
        focal = 1.0 / math.tan(math.radians(float(camera.fov)) / 2.0)
        aspect = width / max(height, 1)
        local = np.stack((ndc_x * depth * np.float32(aspect / focal),
                          ndc_y * depth * np.float32(1.0 / focal), -depth), axis=-1)
        eye, view = scene3d._view_basis(camera)
        points = eye + local.reshape(-1, 3) @ view
        start_xy, start_z = scene3d.project(camera_start, width, height, points)
        end_xy, end_z = scene3d.project(camera_end, width, height, points)
        field = (end_xy - start_xy).reshape(data.height, data.width, 2).astype(np.float32)
        valid = (np.isfinite(depth) & (depth > max(float(camera.near), 1e-6))
                 & (depth < float(camera.far)) & (start_z.reshape(depth.shape) > camera_start.near)
                 & (end_z.reshape(depth.shape) > camera_end.near))
        field[~valid] = 0.0
        return field

    @staticmethod
    def _window_node(kind, p, source):
        """Position, BlackOutside and AdjustBBox: move or resize the data window, never resample.

        * Position: window and pixels shift together by whole pixels; the display window stays put.
        * BlackOutside: the window grows one pixel on every side and the new ring is 0 (black and
          transparent, the same "no pixel here" every uncovered area already is), so a filter
          downstream reads black at the old edge instead of extending the edge pixel.
        * AdjustBBox: `numpixels` grows the window (zero fill) or, negative, shrinks it (pixels
          outside are dropped); pixels that stay are byte-identical and stay where they were.
          `clip_to_format` then intersects the result with the display window.
        """
        box = source.data
        if kind == "Position":
            box = Region(box.x + int(p["translate_x"]), box.y + int(p["translate_y"]), box.width, box.height)
            return Raster(source.pixels, box, source.display)
        if box.is_empty:
            return source
        if kind == "BlackOutside":
            grown = box.expand(1, 1)
        else:
            grown = box.expand(int(p["numpixels"]), int(p["numpixels"]))
            if grown.is_empty:
                grown = Region(box.x, box.y, 0, 0)
            if p.get("clip_to_format", 0):
                grown = grown.intersect(source.display)
        return Raster(source.fit(grown), grown, source.display)

    @staticmethod
    def _filter_window(kind, p, source):
        """The rectangle a filter's own output covers, before any mask/mix blending."""
        if kind == "Crop":
            # Crop's whole meaning is "the output is this rectangle". Shrinking the data window
            # here is what stops every downstream node from computing over discarded area.
            return source.data.intersect(Region(int(p["x"]), int(p["y"]),
                                                int(p["width"]), int(p["height"])))
        if kind in ("Transform", "Tracker", "Stabilize"):
            return Evaluator._transformed_window(source.data, p)
        if kind == "CornerPin":
            return Evaluator._cornerpin_window(source.data, p)
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
        matrix = Evaluator._transform_forward_matrix(p)
        xs, ys = [], []
        for px, py in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            q = matrix @ np.array([px, py, 1.0])
            xs.append(q[0] / q[2]); ys.append(q[1] / q[2])
        # Reconstruction reach of the filter, in destination pixels. Rounding outward and adding
        # the support keeps every pixel the resampler can actually write inside the window; a
        # window one pixel short would clip a rotated edge and look like a rendering bug.
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        reach = float(np.linalg.norm(matrix[:2, :2], ord=2))
        margin = int(math.ceil(support * max(1.0, reach)))
        left, top = math.floor(min(xs)) - margin, math.floor(min(ys)) - margin
        right, bottom = math.ceil(max(xs)) + margin, math.ceil(max(ys)) + margin
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _transform_forward_matrix(p):
        """Nuke-style 2D affine matrix shared by pixel sampling and data-window bounds."""
        angle = math.radians(float(p.get("rotate", 0.0)))
        c, s = math.cos(angle), math.sin(angle)
        sx = float(p.get("scale_x", p.get("scale", 1.0)))
        sy = float(p.get("scale_y", p.get("scale", 1.0)))
        if p.get("scale_mode", "uniform") != "xy":
            sx = sy = float(p.get("scale", 1.0))
        kx, ky = float(p.get("skew_x", 0.0)), float(p.get("skew_y", 0.0))
        skew_x = np.array([[1.0, kx], [0.0, 1.0]])
        skew_y = np.array([[1.0, 0.0], [ky, 1.0]])
        skew = skew_y @ skew_x if p.get("skew_order", "XY") == "XY" else skew_x @ skew_y
        linear = np.array([[c, -s], [s, c]]) @ skew @ np.diag([sx, sy])
        cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
        tx, ty = float(p.get("translate_x", 0.0)), float(p.get("translate_y", 0.0))
        forward = np.eye(3)
        forward[:2, :2] = linear
        forward[:2, 2] = [cx + tx, cy + ty] - linear @ np.array([cx, cy])
        if p.get("invert", 0):
            forward = np.linalg.inv(forward)
        return forward

    @staticmethod
    def _solve_homography(src_pts, dst_pts):
        """The 3x3 projective matrix mapping four `src_pts` onto four `dst_pts` exactly.

        Standard 4-point DLT with h33 fixed at 1: eight linear equations (two per
        correspondence) for the eight remaining unknowns, solved exactly rather than by least
        squares, since four correspondences fully determine a projective map. `to == from`
        (CornerPin's own default) solves to the identity matrix, which is the brief's own
        worked example.
        """
        a = np.zeros((8, 8), dtype=np.float64)
        b = np.zeros(8, dtype=np.float64)
        for i, ((x, y), (u, v)) in enumerate(zip(src_pts, dst_pts)):
            a[2 * i] = [x, y, 1, 0, 0, 0, -x * u, -y * u]
            b[2 * i] = u
            a[2 * i + 1] = [0, 0, 0, x, y, 1, -x * v, -y * v]
            b[2 * i + 1] = v
        h = np.linalg.solve(a, b)
        return np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]], [h[6], h[7], 1.0]])

    @staticmethod
    def _cornerpin_forward_matrix(p):
        """The homography mapping a pre-warp canvas point to its post-warp destination point.

        `direction` "forward" (Nuke's default) warps the "from" quad onto the "to" quad;
        "inverse" swaps which quad is the pre-warp side, by using the same solved homography's
        inverse instead. Sharing this one function between the kernel (which needs its inverse,
        destination -> source) and `_cornerpin_window` (which needs it forward, source ->
        destination) is what keeps the two from drifting apart the way `_transform` and
        `_transformed_window` are kept in sync by hand -- see that pair's own docstring.
        """
        homography = Evaluator._solve_homography(
            [(p["from1_x"], p["from1_y"]), (p["from2_x"], p["from2_y"]),
             (p["from3_x"], p["from3_y"]), (p["from4_x"], p["from4_y"])],
            [(p["to1_x"], p["to1_y"]), (p["to2_x"], p["to2_y"]),
             (p["to3_x"], p["to3_y"]), (p["to4_x"], p["to4_y"])])
        return homography if p["direction"] == "forward" else np.linalg.inv(homography)

    @staticmethod
    def _cornerpin_window(box, p):
        """Forward-map the corners of `box` and take the bounding rectangle.

        Exact for a proper (non-self-intersecting, horizon-clear) quad: a projective transform
        maps straight edges to straight edges, so a rectangle's extrema are still at its four
        mapped corners, the same reasoning `_transformed_window` uses for the affine case.
        """
        if box.is_empty:
            return box
        forward = Evaluator._cornerpin_forward_matrix(p)
        xs, ys = [], []
        for px, py in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            denom = forward[2, 0] * px + forward[2, 1] * py + forward[2, 2]
            denom = denom if abs(denom) > 1e-9 else 1e-9
            xs.append((forward[0, 0] * px + forward[0, 1] * py + forward[0, 2]) / denom)
            ys.append((forward[1, 0] * px + forward[1, 1] * py + forward[1, 2]) / denom)
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        left, top = math.floor(min(xs)) - support, math.floor(min(ys)) - support
        right, bottom = math.ceil(max(xs)) + support, math.ceil(max(ys)) + support
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _run_row_chunked(kind, p, source, mask, mix, out: Region, frame, cancel, progress=None):
        """`_filtered_pixels` + `_apply_mask_mix`, in `_CANCEL_CHUNK_ROWS`-row bands.

        Only called for `_ROW_CHUNKED_MASK_MIX_KINDS`, whose kernels are pointwise (each output
        pixel depends on nothing but the same input pixel), so slicing `out` into row bands and
        computing each independently is pixel-identical to computing it as one call -- the same
        contract `tileexec.py`'s tiling of these same kinds already relies on. Raises `Cancelled`
        between bands rather than after the whole rectangle, which is the point: a caller that set
        `cancel` gets to stop after at most one band's worth of work, not the full frame's.

        `progress`, when given the owning `Evaluator`, has its `row_chunks` counter bumped after
        each band -- real, waitable progress a test (or a future progress bar) can synchronise on,
        the same way the existing cancellation tests already wait on `misses`/`cache.misses`.
        """
        pixels = np.empty((out.height, out.width, 4), dtype=np.float32)
        for band_y in range(0, out.height, _CANCEL_CHUNK_ROWS):
            if cancel.is_set():
                raise Cancelled()
            band_height = min(_CANCEL_CHUNK_ROWS, out.height - band_y)
            band = Region(out.x, out.y + band_y, out.width, band_height)
            filtered = Evaluator._filtered_pixels(kind, p, source, band, frame)
            pixels[band_y:band_y + band_height] = Evaluator._apply_mask_mix(
                source.fit(band), filtered, None if mask is None else mask.fit(band), mix)
            if progress is not None:
                progress.row_chunks += 1
        return pixels

    @staticmethod
    def _filtered_pixels(kind, p, source, out: Region, frame=None):
        """The filter's result, evaluated over exactly the rectangle `out`."""
        if kind in ("OCIOColorspace", "OCIODisplay", "OCIOFileTransform", "OCIOLookTransform", "OCIOLogConvert", "Colorspace"):
            from .ocio_nodes import transform
            return transform(kind, p, source.fit(out))
        if kind == "Crop":
            pixels = np.zeros((out.height, out.width, 4), dtype=np.float32)
            keep = out.intersect(source.data).intersect(
                Region(int(p["x"]), int(p["y"]), int(p["width"]), int(p["height"])))
            if not keep.is_empty:
                pixels[keep.y - out.y:keep.bottom - out.y,
                       keep.x - out.x:keep.right - out.x] = source.fit(keep)
            return pixels
        if kind in ("Transform", "Tracker", "Stabilize"):
            return Evaluator._transform(source.pixels, p["translate_x"], p["translate_y"], p["rotate"],
                                        p["scale"], p["center_x"], p["center_y"], p["filter"],
                                        src_box=source.data, dst_box=out, params=p)
        if kind == "CornerPin":
            return Evaluator._cornerpin(source.pixels, p, src_box=source.data, dst_box=out)
        if kind == "Grade":
            return Evaluator._grade(source.fit(out), p)
        if kind == "Histogram":
            return Evaluator._histogram_levels(source.fit(out), p)
        if kind == "HistEQ":
            return Evaluator._hist_eq(source.fit(out), p)
        if kind == "ColorCorrect":
            return Evaluator._color_correct(source.fit(out), p)
        if kind == "Blur":
            return Evaluator._blur(source.fit(out), p)
        if kind == "Invert":
            return Evaluator._invert(source.fit(out), p)
        if kind == "Clamp":
            return Evaluator._clamp(source.fit(out), p)
        if kind == "Multiply":
            return Evaluator._channel_multiply(source.fit(out), p)
        if kind == "Add":
            return Evaluator._channel_add(source.fit(out), p)
        if kind == "Gamma":
            return Evaluator._channel_gamma(source.fit(out), p)
        if kind == "Saturation":
            return Evaluator._saturation(source.fit(out), p)
        if kind == "Exposure":
            return Evaluator._exposure(source.fit(out), p)
        if kind == "HueCorrect":
            return Evaluator._hue_correct(source.fit(out), p)
        if kind == "ColorLookup":
            return Evaluator._color_lookup(source.fit(out), p)
        if kind == "ColorMatrix":
            return Evaluator._color_matrix(source.fit(out), p)
        if kind in ("Log2Lin", "PLogLin", "CrossTalk", "Toe"):
            return Evaluator._kernel(kind, p, [source.fit(out)], frame, origin=(out.x, out.y))
        if kind == "Keyer":
            return Evaluator._keyer(source.fit(out), p)
        if kind == "HueKeyer":
            return Evaluator._hue_keyer(source.fit(out), p)
        if kind == "ChromaKeyer":
            return Evaluator._chroma_keyer(source.fit(out), p)
        if kind == "IBKColor":
            return Evaluator._ibk_color(source.fit(out), p)
        if kind == "Erode":
            return Evaluator._erode(source.fit(out), p)
        if kind == "Dilate":
            return Evaluator._dilate(source.fit(out), p)
        if kind == "Median":
            return Evaluator._median(source.fit(out), p)
        if kind == "Sharpen":
            return Evaluator._sharpen(source.fit(out), p)
        if kind in ("Matrix", "Laplacian"):
            return Evaluator._matrix(source.fit(out), p, laplacian=kind == "Laplacian")
        if kind == "EdgeDetect": return Evaluator._edge_detect(source.fit(out), p)
        if kind == "Emboss": return Evaluator._emboss(source.fit(out), p)
        if kind == "BumpBoss": return Evaluator._bump_boss(source.fit(out), p)
        if kind == "ErodeFilter": return Evaluator._erode_filter(source.fit(out), p)
        if kind == "Glow":
            return Evaluator._glow(source.fit(out), p)
        if kind == "Soften":
            return Evaluator._soften(source.fit(out), p)
        if kind == "Defocus":
            return Evaluator._defocus(source.fit(out), p)
        if kind == "Bilateral":
            return Evaluator._bilateral(source.fit(out), p)
        if kind == "Denoise":
            # Practical spatial bilateral denoising; temporal sampling is described separately
            # because the evaluator currently has no neighbouring-frame cache contract.
            return Evaluator._bilateral(source.fit(out), {"spatial_size": 2.0 + 5.0 * float(p.get("denoise_strength", 0.2)),
                                                          "colour_sigma": max(0.01, float(p.get("denoise_strength", 0.2)))})
        if kind == "DegrainSimple":
            return Evaluator._degrain_simple(source.fit(out), p)
        if kind == "DropShadow":
            return Evaluator._drop_shadow(source.fit(out), p)
        if kind == "EdgeBlur":
            return Evaluator._edge_blur(source.fit(out), p)
        if kind == "EdgeExtend":
            return Evaluator._edge_extend(source.fit(out), p)
        if kind == "Dither":
            return Evaluator._dither(source.fit(out), p, origin=(out.x, out.y))
        if kind == "Grain":
            return Evaluator._grain(source.fit(out), p, origin=(out.x, out.y), frame=frame)
        if kind == "Posterize":
            return Evaluator._posterize(source.fit(out), p)
        if kind == "SoftClip":
            return Evaluator._softclip(source.fit(out), p)
        if kind == "HSVTool":
            return Evaluator._hsv_tool(source.fit(out), p)
        if kind == "DirBlur":
            # Radial and zoom are centred on a canvas point; the kernel sees only an array, so the
            # centre is handed over relative to this rectangle's own corner.
            local = dict(p, center_x=float(p.get("center_x", 0.0)) - out.x,
                         center_y=float(p.get("center_y", 0.0)) - out.y)
            return Evaluator._dirblur(source.fit(out), local)
        if kind == "Mirror":
            return Evaluator._mirror(source.fit(out), p)
        raise ValueError(f"No windowed filter for {kind}")

    @staticmethod
    def _decimate(pixels, tier):
        """Area-average downscale by an integer factor, matching ceil(n / tier) extents.

        Area averaging rather than point sampling because a proxy is judged by whether it predicts
        the full-resolution result; point sampling aliases a detailed plate into a different image
        and would make the tier lie about the shot.

        Averages by adding `tier` strided views and scaling once, rather than a single
        `reshape(..., tier, ..., tier, ...).mean(axis=(1, 3))` call: the reshape-and-multi-axis-
        mean forces NumPy to walk the array in an access pattern that defeats its fast reduction
        loops, measured ~8x slower than this for both tier 2 and tier 4 on a 4K frame
        (docs/BENCHMARKS-v0.17-playback.md) despite doing the identical arithmetic (a sum of the
        same `tier * tier` input values divided by `tier * tier`, just accumulated in a
        different order, so results agree with the old implementation to float32 rounding).
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
        rows_reduced = pixels[0::tier]
        for offset in range(1, tier):
            rows_reduced = rows_reduced + pixels[offset::tier]
        rows_reduced *= np.float32(1.0 / tier)
        columns_reduced = rows_reduced[:, 0::tier]
        for offset in range(1, tier):
            columns_reduced = columns_reduced + rows_reduced[:, offset::tier]
        columns_reduced *= np.float32(1.0 / tier)
        return columns_reduced.astype(np.float32)

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
    def _kernel(kind, p, inputs, frame=None, data=None, origin=(0, 0)):
        if kind in ("OCIOColorspace", "OCIODisplay", "OCIOFileTransform", "OCIOLookTransform", "OCIOLogConvert", "Colorspace"):
            from .ocio_nodes import transform
            filtered = transform(kind, p, inputs[0])
            return Evaluator._apply_mask_mix(inputs[0], filtered,
                mask=inputs[1] if len(inputs) > 1 else None, mix=p.get("mix", 1.0))
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
        if kind in ("Viewer", "Write", "NoOp", "PostageStamp", "Output", "Profile"):
            # Write is a tap, not a transform: rendering it is an explicit action, and the pixels
            # continue downstream untouched so parking one mid-branch changes nothing.
            return inputs[0]
        if kind == "Grade":
            filtered = Evaluator._grade(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("Log2Lin", "PLogLin", "CrossTalk", "Toe", "Expression"):
            if kind == "Expression":
                second = inputs[1] if len(inputs) > 1 else None
                mask = inputs[2] if len(inputs) > 2 else None
                filtered = Evaluator._expression(inputs[0], second, p, frame, origin)
            else:
                op = {"Log2Lin": Evaluator._log2lin, "PLogLin": Evaluator._ploglin,
                      "CrossTalk": Evaluator._crosstalk, "Toe": Evaluator._toe}[kind]
                filtered = op(inputs[0], p)
                mask = inputs[1] if len(inputs) > 1 else None
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=mask, mix=p.get("mix", 1.0))
        if kind == "ColorCorrect":
            filtered = Evaluator._color_correct(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Blur":
            filtered = Evaluator._blur(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("Transform", "Tracker", "Stabilize"):
            filtered = Evaluator._transform(inputs[0], p["translate_x"], p["translate_y"], p["rotate"],
                                              p["scale"], p["center_x"], p["center_y"], p["filter"],
                                              params=p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Crop":
            filtered = Evaluator._crop(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Invert":
            filtered = Evaluator._invert(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Clamp":
            filtered = Evaluator._clamp(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Multiply":
            filtered = Evaluator._channel_multiply(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Add":
            filtered = Evaluator._channel_add(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Gamma":
            filtered = Evaluator._channel_gamma(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Saturation":
            filtered = Evaluator._saturation(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Keyer":
            filtered = Evaluator._keyer(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "HueKeyer":
            filtered = Evaluator._hue_keyer(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ChromaKeyer":
            filtered = Evaluator._chroma_keyer(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ScreenKeyer":
            # Slots: image, inside, outside, clean, mask.
            extra = [inputs[i] if len(inputs) > i else None for i in (1, 2, 3, 4)]
            for name, other in zip(("inside", "outside", "clean", "mask"), extra):
                if other is not None and other.shape != inputs[0].shape:
                    raise ValueError(f"ScreenKeyer {name} must match the image's format in M0")
            filtered = Evaluator._screen_keyer(inputs[0], p, inside=extra[0], outside=extra[1], clean=extra[2])
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=extra[3], mix=p.get("mix", 1.0))
        if kind == "IBKColor":
            filtered = Evaluator._ibk_color(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "IBKGizmo":
            fg, plate = inputs[0], inputs[1]
            bg = inputs[2] if len(inputs) > 2 else None
            if fg.shape != plate.shape:
                raise ValueError("IBKGizmo inputs must have matching formats in M0")
            keyed = Evaluator._ibk_gizmo(fg, plate, bg, p)
            return Evaluator._apply_mask_mix(fg, keyed, mask=inputs[3] if len(inputs) > 3 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Erode":
            filtered = Evaluator._erode(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Dilate":
            filtered = Evaluator._dilate(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Median":
            filtered = Evaluator._median(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Sharpen":
            filtered = Evaluator._sharpen(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("Matrix", "Laplacian"):
            filtered = Evaluator._matrix(inputs[0], p, laplacian=kind == "Laplacian")
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("EdgeDetect", "Emboss", "BumpBoss", "ErodeFilter"):
            op = {"EdgeDetect": Evaluator._edge_detect, "Emboss": Evaluator._emboss,
                  "BumpBoss": Evaluator._bump_boss, "ErodeFilter": Evaluator._erode_filter}[kind]
            filtered = op(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Glow":
            filtered = Evaluator._glow(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Exposure":
            filtered = Evaluator._exposure(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "HueCorrect":
            filtered = Evaluator._hue_correct(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ColorLookup":
            filtered = Evaluator._color_lookup(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "ColorMatrix":
            filtered = Evaluator._color_matrix(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Soften":
            filtered = Evaluator._soften(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Defocus":
            filtered = Evaluator._defocus(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "DropShadow":
            filtered = Evaluator._drop_shadow(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("EdgeBlur", "EdgeExtend", "Dither"):
            filtered = getattr(Evaluator, {"EdgeBlur": "_edge_blur", "EdgeExtend": "_edge_extend",
                                           "Dither": "_dither"}[kind])(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("Grain", "Posterize", "SoftClip", "HSVTool", "Flare", "Glint", "Sparkles", "GodRays", "VolumeRays", "ScannedGrain"):
            if kind == "Grain":
                filtered = Evaluator._grain(inputs[0], p, origin=origin, frame=frame)
            else:
                if kind == "Flare": filtered = Evaluator._flare(inputs[0], p, origin)
                elif kind == "Glint": filtered = Evaluator._glint(inputs[0], p)
                elif kind == "Sparkles": filtered = Evaluator._sparkles(inputs[0], p, frame, origin)
                elif kind in ("GodRays", "VolumeRays"):
                    filtered = Evaluator._godrays(inputs[0], p, origin,
                        inputs[1] if kind == "VolumeRays" and len(inputs) > 1 else None)
                elif kind == "ScannedGrain": filtered = Evaluator._scanned_grain(inputs[0], inputs[1], p, frame)
                else: filtered = getattr(Evaluator, {"Posterize": "_posterize", "SoftClip": "_softclip", "HSVTool": "_hsv_tool"}[kind])(inputs[0], p)
            mask_index = 2 if kind == "ScannedGrain" or kind == "VolumeRays" else 1
            mask = inputs[mask_index] if len(inputs) > mask_index else None
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=mask,
                                              mix=p.get("mix", 1.0))
        if kind == "AddMix":
            a, b = inputs[0], inputs[1]
            if a.shape != b.shape:
                raise ValueError("AddMix inputs must have matching formats in M0")
            return Evaluator._merge_gated("over", Evaluator._premultiply(a), b, p.get("mix", 1.0),
                                          inputs[2] if len(inputs) > 2 else None)
        if kind == "CopyRectangle":
            a, b = inputs[0], inputs[1]
            if a.shape != b.shape:
                raise ValueError("CopyRectangle inputs must have matching formats in M0")
            copied = Evaluator._copy_rectangle(a, b, p, origin=origin)
            return Evaluator._apply_mask_mix(b, copied, mask=inputs[2] if len(inputs) > 2 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Blend":
            layers = [(i, image) for i, image in enumerate(inputs[:16]) if image is not None]
            if not layers:
                raise ValueError("Blend: connect at least one input")
            blended = Evaluator._blend(layers, p)
            mask = inputs[16] if len(inputs) > 16 else None
            channel = p.get("mask_channel", "alpha")
            out = Evaluator._apply_mask_mix(layers[0][1], blended, mask=mask, mix=p.get("mix", 1.0),
                                            mask_channel=channel)
            if p.get("inject") and mask is not None:
                # Nuke's inject: the matte itself becomes the output's alpha, whatever `mix` says.
                out = out.copy()
                out[..., 3:4] = Evaluator._mask_matte(mask, channel)
            return out
        if kind == "LightWrap":
            fg, bg = inputs[0], inputs[1]
            if fg.shape != bg.shape:
                raise ValueError("LightWrap inputs must have matching formats in M0")
            wrapped = Evaluator._light_wrap(fg, bg, p)
            return Evaluator._apply_mask_mix(fg, wrapped, mask=inputs[2] if len(inputs) > 2 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "DirBlur":
            filtered = Evaluator._dirblur(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind == "Mirror":
            filtered = Evaluator._mirror(inputs[0], p)
            return Evaluator._apply_mask_mix(inputs[0], filtered, mask=inputs[1] if len(inputs) > 1 else None,
                                              mix=p.get("mix", 1.0))
        if kind in ("Dissolve", "TimeDissolve"):
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Dissolve inputs must have matching formats in M0")
            blended = Evaluator._dissolve(a, b, p)
            return Evaluator._apply_mask_mix(b, blended, mask=mask, mix=p.get("mix", 1.0))
        if kind == "Keymix":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Keymix inputs must have matching formats in M0")
            key_mask = mask
            if key_mask is not None and p.get("invert_mask"):
                key_mask = np.concatenate([key_mask[..., :3], 1 - key_mask[..., 3:4]], axis=-1)
            return Evaluator._apply_mask_mix(b, a, mask=key_mask, mix=p.get("mix", 1.0))
        if kind == "Copy":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Copy inputs must have matching formats in M0")
            copied = Evaluator._copy_channels(a, b, p)
            return Evaluator._apply_mask_mix(b, copied, mask=mask, mix=p.get("mix", 1.0))
        if kind == "ChannelMerge":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("ChannelMerge inputs must have matching formats in M0")
            merged = Evaluator._channel_merge(a, b, p)
            return Evaluator._apply_mask_mix(b, merged, mask=mask, mix=p.get("mix", 1.0))
        if kind == "Difference":
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Difference inputs must have matching formats in M0")
            keyed = Evaluator._difference_key(a, b, p)
            return Evaluator._apply_mask_mix(b, keyed, mask=mask, mix=p.get("mix", 1.0))
        if kind == "Roto":
            from . import roto
            return roto.rasterise(data or [], p["width"], p["height"], bool(p.get("invert", 0)))
        if kind == "RotoPaint":
            from . import paint
            base = inputs[0]
            if base is None:
                raise ValueError("RotoPaint: connect an image")
            source = inputs[1] if len(inputs) > 1 else None
            return paint.rasterise(base, data or [], frame, source=source)
        if kind == "ChannelShuffle":
            a, b = inputs[0], (inputs[1] if len(inputs) > 1 else None)
            index = {"r": 0, "g": 1, "b": 2, "a": 3}
            h, w = a.shape[:2]

            def pick(name):
                if name in ("0", "1"):
                    return np.full((h, w, 1), float(name), dtype=np.float32)
                which, channel = name.split(".")
                if which == "B":
                    if b is None:
                        # Black would be a plausible-looking result for a wiring mistake, and the
                        # artist would find it in a review rather than here.
                        raise ValueError(f"ChannelShuffle routes {name} but input B is not connected")
                    return b[..., index[channel]:index[channel] + 1]
                return a[..., index[channel]:index[channel] + 1]
            return np.concatenate([pick(p["out_red"]), pick(p["out_green"]),
                                   pick(p["out_blue"]), pick(p["out_alpha"])], axis=2).astype(np.float32)
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
            a, b = inputs[0], inputs[1]
            mask = inputs[2] if len(inputs) > 2 else None
            if a.shape != b.shape:
                raise ValueError("Merge inputs must have matching formats in M0")
            return Evaluator._merge_gated(p.get("operation", "over"), a, b, p["mix"], mask)
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
    def _merge_gated(op, a, b, mix, mask=None):
        """Merge A onto B, gated by `mix` and, when wired, by the mask's alpha per pixel.

        gate = mix * mask.a (a scalar `mix` when the mask is unwired). `over` scales A by the gate
        before compositing; every other operation blends its result against B by the gate. With no
        mask this is byte-identical to the pre-mask Merge, and where mask.a is 0 the output is B.
        """
        if mask is not None and mask.shape[:2] != b.shape[:2]:
            raise ValueError("Merge mask must match the merged format")
        gate = np.float32(mix) if mask is None else np.float32(mix) * mask[..., 3:4]
        if op == "over":
            # Keep the v0.3.0 mix behaviour byte-identical: scale A by mix, then standard over.
            return a * gate + b * (1 - (a[..., 3:4] * gate))
        full = Evaluator._merge_op(op, a, b)
        return full * gate + b * (1 - gate)

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
        if op == "average":
            return (a + b) * np.float32(0.5)
        if op == "from":
            return b - a
        if op == "hypot":
            return np.sqrt(a * a + b * b)
        return Evaluator._merge_op_extra(op, a, b, aa, ba)

    @staticmethod
    def _merge_op_extra(op, a, b, aa, ba):
        """The eleven Nuke operations added in step 3a, shared by `_merge_op` (aa/ba are the
        alpha slices) and `_channel_merge_op` (aa/ba are the channel values themselves).

        Formulas follow Foundry's documented merge algorithms. Division convention: every
        divisor with magnitude <= 1e-6 is treated as unusable and the guarded term is replaced
        by the limit chosen below, so zero and HDR inputs always give finite results.
        disjoint-over: the B(1-a)/b term is 0 where b is unusable. conjoint-over: A alone where
        b is unusable. geometric: 0 where A+B is unusable. color-dodge: 1 where 1-A is unusable
        and B > 0 (else 0). color-burn: 0 where A is unusable, except 1 when B >= 1.
        soft-light keeps Foundry's documented A*B < 1 branch test on the raw values.
        """
        eps = np.float32(1e-6)
        one = np.float32(1.0)
        zero = np.float32(0.0)
        if op == "matte":
            return a * aa + b * (1 - aa)
        if op == "disjoint-over":
            safe = np.where(np.abs(ba) > eps, ba, one)
            tail = np.where(np.abs(ba) > eps, b * (1 - aa) / safe, zero)
            return np.where(aa + ba < 1, a + b, a + tail).astype(np.float32)
        if op == "conjoint-over":
            safe = np.where(np.abs(ba) > eps, ba, one)
            ratio = np.where(np.abs(ba) > eps, aa / safe, one)
            return np.where((aa > ba) | (np.abs(ba) <= eps), a, a + b * (1 - ratio)).astype(np.float32)
        if op == "copy":
            return a.copy()
        if op == "exclusion":
            return a + b - 2 * a * b
        if op == "geometric":
            total = a + b
            safe = np.where(np.abs(total) > eps, total, one)
            return np.where(np.abs(total) > eps, 2 * a * b / safe, zero).astype(np.float32)
        if op == "overlay":
            return np.where(b < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b)).astype(np.float32)
        if op == "hard-light":
            return np.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b)).astype(np.float32)
        if op == "soft-light":
            ab = a * b
            return np.where(ab < 1, b * (2 * a + b * (1 - ab)), 2 * ab).astype(np.float32)
        if op == "color-dodge":
            gap = 1 - a
            safe = np.where(gap > eps, gap, one)
            limit = np.where(b > 0, one, zero)
            return np.where(gap > eps, b / safe, limit).astype(np.float32)
        if op == "color-burn":
            safe = np.where(a > eps, a, one)
            limit = np.where(b >= 1, one, zero)
            return np.where(a > eps, 1 - (1 - b) / safe, limit).astype(np.float32)
        raise ValueError(f"Unknown merge operation: {op}")

    @staticmethod
    def _channel_merge_op(op, a, b):
        """Single-channel counterpart of `_merge_op`.

        ChannelMerge's A and B are one selected scalar channel each, not full RGBA, so there is no
        separate alpha to key off. Nuke's own convention (and the one this mirrors) is that each
        side's own value stands in for its alpha, which is why `aa` and `ba` are just `a` and `b`
        rather than a fourth channel slice.
        """
        aa, ba = a, b
        if op == "over":
            return a + b * (1 - aa)
        if op == "under":
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
            safe = np.where(np.abs(b) > 1e-6, b, np.float32(1.0))
            return np.where(np.abs(b) > 1e-6, a / safe, np.float32(0.0)).astype(np.float32)
        if op == "mask":
            return b * aa
        if op == "stencil":
            return b * (1 - aa)
        if op == "in":
            return a * ba
        if op == "out":
            return a * (1 - ba)
        if op == "atop":
            return a * ba + b * (1 - aa)
        if op == "xor":
            return a * (1 - ba) + b * (1 - aa)
        if op == "average":
            return (a + b) * np.float32(0.5)
        if op == "from":
            return b - a
        if op == "hypot":
            return np.sqrt(a * a + b * b)
        return Evaluator._merge_op_extra(op, a, b, aa, ba)

    # Channel sets for the "channels" knob on Invert/Clamp/Multiply/Add/Gamma. "rgba" reaches
    # alpha too; the default "rgb" leaves it untouched, matching Nuke's own default.
    _CHANNEL_SETS = {"rgb": (0, 1, 2), "rgba": (0, 1, 2, 3), "alpha": (3,)}

    @staticmethod
    def _invert(image, p):
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = 1.0 - image[..., c]
        return out

    @staticmethod
    def _clamp(image, p):
        out = image.copy()
        lo = p.get("minimum", 0.0) if p.get("clamp_min", 1) else -np.inf
        hi = p.get("maximum", 1.0) if p.get("clamp_max", 1) else np.inf
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = np.clip(image[..., c], lo, hi)
        return out

    @staticmethod
    def _channel_multiply(image, p):
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = image[..., c] * p.get("multiply", 1.0)
        return out

    @staticmethod
    def _channel_add(image, p):
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = image[..., c] + p.get("offset", 0.0)
        return out

    @staticmethod
    def _channel_gamma(image, p):
        out = image.copy()
        g = p.get("gamma", 1.0)
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            v = image[..., c]
            # sign * |x|^(1/gamma), as ColorCorrect's gamma does: avoids raising a negative HDR
            # value to a fractional power.
            out[..., c] = np.sign(v) * np.abs(v) ** (1.0 / g)
        return out

    @staticmethod
    def _saturation(image, p):
        # Unlike Multiply/Add/Gamma, Saturation has no channels knob in Nuke: it is inherently an
        # RGB-luma operation and always leaves alpha untouched.
        rgb = image[..., :3]
        luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
        out_rgb = luma + (rgb - luma) * p.get("saturation", 1.0)
        return np.concatenate([out_rgb, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _range_ramp(value, a, b, c, d):
        """Nuke Keyer's four-point range: 0 at or below `a`, ramping to 1 between `a` and `b`, 1
        from `b` through `c`, ramping back to 0 between `c` and `d`, 0 at or above `d`.

        Checked as a waterfall of exclusive conditions (each `elif` only reached once the ones
        above it are false) rather than independent masks, so a degenerate zero-width segment
        (a == b, or c == d) falls straight through to its neighbour instead of two masks
        disagreeing about who owns the boundary point. `np.where(b > a, b - a, 1.0)` (and the `d`/
        `c` equivalent) only guards the division: when the ramp is degenerate that branch's mask
        is empty, so the substitute denominator is never actually used.
        """
        rise_span = np.where(b > a, b - a, 1.0)
        fall_span = np.where(d > c, d - c, 1.0)
        return np.where(value <= a, 0.0,
                        np.where(value < b, (value - a) / rise_span,
                                 np.where(value <= c, 1.0,
                                          np.where(value < d, 1.0 - (value - c) / fall_span, 0.0))))

    @staticmethod
    def _keyer(image, p):
        # Alpha only; RGB passes through untouched (Nuke's own Keyer never recolours, it only
        # writes a new matte for whatever's downstream -- typically a Premult -- to use).
        rgb = image[..., :3]
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        op = p.get("keyer_operation", "luminance")
        if op == "luminance":
            value = 0.2126 * r + 0.7152 * g + 0.0722 * b
        elif op == "red":
            value = r
        elif op == "green":
            value = g
        elif op == "blue":
            value = b
        elif op == "min":
            value = np.minimum(np.minimum(r, g), b)
        elif op == "max":
            value = np.maximum(np.maximum(r, g), b)
        elif op == "saturation":
            mx = np.maximum(np.maximum(r, g), b)
            mn = np.minimum(np.minimum(r, g), b)
            value = np.where(mx > 0, (mx - mn) / np.where(mx > 0, mx, 1.0), 0.0)
        else:
            raise ValueError(f"Unknown Keyer operation: {op}")
        alpha = Evaluator._range_ramp(value, p.get("range_a", 0.0), p.get("range_b", 0.0),
                                      p.get("range_c", 1.0), p.get("range_d", 1.0))
        if p.get("invert"):
            alpha = 1.0 - alpha
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    # --- Keyer-menu group K1: ChromaKeyer, IBKColor, IBKGizmo ------------------------------------

    @staticmethod
    def _luma(rgb):
        return 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]

    @staticmethod
    def _chroma_key_alpha(rgb, p):
        """Alpha of a chroma key: the distance of a pixel's brightness-normalised colour from the
        key colour's, in units of the key colour's own distance from neutral grey (so 0 is the
        screen, about 1 is a grey, above 1 is a hue on the far side of it), ramped from 0 at
        `key_tolerance` to 1 at `key_tolerance + key_softness`. Normalising by luminance makes a
        shadowed screen key like a lit one; `luma_gain` puts a share of the brightness difference
        back. Pixels darker than `shadow_level` (unreliable chroma) or brighter than
        `highlight_level` are pulled toward opaque over a short ramp."""
        key = np.array([p.get("key_red", 0.1), p.get("key_green", 0.8), p.get("key_blue", 0.2)],
                       dtype=np.float32)
        key_y = max(float(0.2126 * key[0] + 0.7152 * key[1] + 0.0722 * key[2]), 1e-3)
        n_key = key / key_y
        scale = max(float(np.linalg.norm(n_key - 1.0)), 1e-3)
        y = Evaluator._luma(rgb)
        normalised = rgb / np.maximum(y, 1e-3)
        dist = np.linalg.norm(normalised - n_key, axis=-1, keepdims=True) / scale
        dist = dist + float(p.get("luma_gain", 0.0)) * np.abs(y - key_y) / key_y
        softness = max(float(p.get("key_softness", 0.3)), 1e-6)
        alpha = np.clip((dist - float(p.get("key_tolerance", 0.35))) / softness, 0.0, 1.0)
        shadow = float(p.get("shadow_level", 0.02))
        if shadow > 0.0:
            alpha = alpha + (1.0 - alpha) * np.clip(1.0 - y / shadow, 0.0, 1.0)
        highlight = float(p.get("highlight_level", 1000000.0))
        alpha = alpha + (1.0 - alpha) * np.clip((y - highlight) / max(0.1 * highlight, 1e-6), 0.0, 1.0)
        return alpha.astype(np.float32)

    @staticmethod
    def _screen_channels(dominant):
        """The dominant channel index and the two others, in RGB order."""
        others = [c for c in range(3) if c != dominant]
        return others[0], others[1]

    @staticmethod
    def _despill(rgb, key, bias):
        """Screen-colour suppression: the key colour's dominant channel is limited to a weighted
        mean of the other two, `(1 - bias)` of the first (in RGB order) and `bias` of the second."""
        dominant = int(np.argmax(key))
        first, second = Evaluator._screen_channels(dominant)
        limit = (1.0 - bias) * rgb[..., first] + bias * rgb[..., second]
        out = rgb.copy()
        out[..., dominant] = np.minimum(rgb[..., dominant], limit)
        return out

    @staticmethod
    def _chroma_keyer(image, p):
        rgb = image[..., :3]
        alpha = Evaluator._chroma_key_alpha(rgb, p)
        if p.get("invert"):
            alpha = 1.0 - alpha
        if p.get("despill", 1):
            key = np.array([p.get("key_red", 0.1), p.get("key_green", 0.8), p.get("key_blue", 0.2)],
                           dtype=np.float32)
            rgb = Evaluator._despill(rgb, key, float(p.get("despill_bias", 0.5)))
        if p.get("premultiply"):
            rgb = rgb * alpha
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _bias_weight(colour, dominant, fallback):
        """The share of the second of the two non-screen channels in a bias colour (their weighted
        mean's weight on it); `fallback` for a colour with no red, green or blue to weigh."""
        first, second = Evaluator._screen_channels(dominant)
        total = float(colour[first] + colour[second])
        return float(colour[second]) / total if total > 1e-6 else fallback

    @staticmethod
    def _bias_colour(p, prefix):
        return np.array([p.get(f"{prefix}_red", 0.5), p.get(f"{prefix}_green", 0.5),
                         p.get(f"{prefix}_blue", 0.5)], dtype=np.float32)

    @staticmethod
    def _screen_matte(rgb, p, clean=None):
        """Screen-difference matte. With the screen colour's dominant channel D and the other two
        channels F, S, a pixel's screen difference is D - ((1 - balance) * F + balance * S); the
        screen saturation is that over the screen colour's own difference, so it is 1 on the
        screen and 0 on any grey. The matte is 1 - (saturation - alpha_bias) * screen_gain, clamped
        to 0..1 (0 on the screen, 1 on the foreground; more gain removes more of the screen)."""
        key = np.array([p.get("screen_red", 0.1), p.get("screen_green", 0.8), p.get("screen_blue", 0.2)],
                       dtype=np.float32)
        dominant = int(np.argmax(key))
        first, second = Evaluator._screen_channels(dominant)
        balance = float(p.get("screen_balance", 0.5))
        if p.get("bias_colours"):
            balance = Evaluator._bias_weight(Evaluator._bias_colour(p, "alpha_bias"), dominant, balance)
        key_diff = max(float(key[dominant] - ((1.0 - balance) * key[first] + balance * key[second])), 1e-4)
        diff = rgb[..., dominant:dominant + 1] - ((1.0 - balance) * rgb[..., first:first + 1]
                                                   + balance * rgb[..., second:second + 1])
        if clean is not None:
            # A clean plate is the screen reference pixel by pixel: where it carries a screen
            # difference the saturation is measured against it, elsewhere against the screen colour.
            plate = clean[..., :3]
            plate_diff = plate[..., dominant:dominant + 1] - ((1.0 - balance) * plate[..., first:first + 1]
                                                               + balance * plate[..., second:second + 1])
            key_diff = np.where(plate_diff > 1e-4, plate_diff, np.float32(key_diff))
        saturation = diff / key_diff - float(p.get("alpha_bias", 0.0))
        return np.clip(1.0 - saturation * float(p.get("screen_gain", 1.0)), 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _screen_clip(matte, p):
        """Clip black and white: at or below `clip_black` the matte is 0, at or above `clip_white`
        it is 1, and between them it is stretched to fill 0..1. `clip_rollback` then returns a share
        of what the clip flattened, weighted by 4m(1-m) so only semi-transparent pixels move."""
        black, white = float(p.get("clip_black", 0.0)), float(p.get("clip_white", 1.0))
        if black == 0.0 and white == 1.0:
            return matte
        clipped = np.clip((matte - black) / max(white - black, 1e-6), 0.0, 1.0)
        rollback = float(p.get("clip_rollback", 0.0))
        if rollback > 0.0:
            clipped = clipped + rollback * (matte - clipped) * 4.0 * matte * (1.0 - matte)
        return clipped.astype(np.float32)

    @staticmethod
    def _screen_matte_final(rgb, p, clean=None, inside=None, outside=None):
        """The finished matte: screen difference, clip and rollback, then shrink/grow (positive
        grows the screen, i.e. minimum filters the matte) and softness (a Gaussian, sigma = size / 3),
        then the garbage mattes: the `inside` input's alpha forces foreground (matte = max) and the
        `outside` input's alpha forces background (matte *= 1 - outside, so outside wins an overlap)."""
        matte = Evaluator._screen_clip(Evaluator._screen_matte(rgb, p, clean), p)
        shrink = float(p.get("screen_shrink", 0.0))
        if abs(shrink) >= 0.5:
            matte = Evaluator._box_extreme(matte, abs(shrink), use_max=(shrink < 0))
        softness = abs(float(p.get("screen_softness", 0.0)))
        if softness >= 0.5:
            radius = int(math.ceil(softness))
            sigma = softness / 3.0
            matte = Evaluator._gaussian_axis(Evaluator._gaussian_axis(matte, radius, sigma, axis=1),
                                             radius, sigma, axis=0)
        matte = np.clip(matte, 0.0, 1.0)
        if inside is not None:
            matte = np.maximum(matte, inside[..., 3:4])
        if outside is not None:
            matte = matte * (1.0 - outside[..., 3:4])
        return np.clip(matte, 0.0, 1.0).astype(np.float32)

    @staticmethod
    def _screen_keyer(image, p, inside=None, outside=None, clean=None):
        """Keylight-style screen-difference keyer (docs/PARITY_2D.md). `view` picks the output:
        final (despilled colour premultiplied by the matte), status (0 background, 1 foreground,
        mid-grey for every pixel the matte leaves neither), screen_matte (the matte as grey and
        alpha) or intermediate (the despilled colour with the input's alpha, no matte)."""
        rgb = image[..., :3]
        matte = Evaluator._screen_matte_final(rgb, p, clean, inside, outside)
        view = p.get("keyer_view", "final")
        if view == "screen_matte":
            return np.concatenate([matte, matte, matte, matte], axis=2).astype(np.float32)
        if view == "status":
            partial = (matte > 0.0) & (matte < 1.0)
            grey = np.where(partial, 0.5, matte).astype(np.float32)
            return np.concatenate([grey, grey, grey, np.ones_like(grey)], axis=2)
        key = np.array([p.get("screen_red", 0.1), p.get("screen_green", 0.8), p.get("screen_blue", 0.2)],
                       dtype=np.float32)
        bias = float(p.get("despill_bias", 0.5))
        if p.get("bias_colours"):
            bias = Evaluator._bias_weight(Evaluator._bias_colour(p, "despill_bias"), int(np.argmax(key)), bias)
        despilled = Evaluator._despill(rgb, key, bias)
        if view == "intermediate":
            return np.concatenate([despilled, image[..., 3:4]], axis=2).astype(np.float32)
        return np.concatenate([despilled * matte, matte], axis=2).astype(np.float32)

    @staticmethod
    def _box3_sum(frame):
        """3x3 box sum with zero fill outside the array (a separable pair of 3-tap passes)."""
        h, w = frame.shape[:2]
        padded = np.pad(frame, ((0, 0), (1, 1), (0, 0)))
        rows = padded[:, 0:w] + padded[:, 1:w + 1] + padded[:, 2:w + 2]
        padded = np.pad(rows, ((1, 1), (0, 0), (0, 0)))
        return padded[0:h] + padded[1:h + 1] + padded[2:h + 2]

    @staticmethod
    def _ibk_color(image, p):
        """A clean screen plate from a blue or green screen. Pixels whose screen channel leads the
        larger of the other two by more than a quarter of itself count as known screen; the known
        set is eroded by `screen_erode` pixels so foreground edges and spill drop out, then each of
        `fill_size` passes gives every unknown pixel next to known ones the mean of its known 3x3
        neighbours (only known screen ever contributes, so the fill never smears foreground). A
        pixel further than `fill_size` from any screen stays unresolved: black, or with
        `patch_black` a dark screen colour at the `darks` level. The plate's screen channel is
        clamped to `darks`..`lights`; alpha is 1."""
        rgb = image[..., :3].astype(np.float32)
        dominant = 2 if p.get("screen_type", "green") == "blue" else 1
        first, second = Evaluator._screen_channels(dominant)
        lead = rgb[..., dominant:dominant + 1] - np.maximum(rgb[..., first:first + 1], rgb[..., second:second + 1])
        known = (lead > 0.25 * rgb[..., dominant:dominant + 1]) & (rgb[..., dominant:dominant + 1] > 0)
        erode = float(p.get("screen_erode", 1.0))
        if erode >= 0.5:
            known = Evaluator._box_extreme(known.astype(np.float32), erode, use_max=False) > 0.5
        plate = np.where(known, rgb, 0.0).astype(np.float32)
        for _ in range(int(p.get("fill_size", 10))):
            if known.all():
                break
            counts = Evaluator._box3_sum(known.astype(np.float32))
            sums = Evaluator._box3_sum(plate)
            new = (~known) & (counts > 0)
            plate = np.where(new, sums / np.maximum(counts, 1.0), plate)
            known = known | new
        darks, lights = float(p.get("darks", 0.05)), float(p.get("lights", 1000.0))
        channel = plate[..., dominant:dominant + 1]
        factor = np.where(channel > 1e-6, np.clip(channel, darks, lights) / np.maximum(channel, 1e-6), 1.0)
        plate = np.where(known, plate * factor, plate)
        if p.get("patch_black", 1):
            patch = np.zeros_like(plate)
            patch[..., dominant] = darks
            plate = np.where(known, plate, patch)
        return np.concatenate([plate, np.ones_like(plate[..., :1])], axis=2).astype(np.float32)

    @staticmethod
    def _ibk_gizmo(fg, plate, bg, p):
        """Colour-difference key of the foreground against a clean plate. With d = screen channel
        minus `red_weight` * red + `blue_green_weight` * (the remaining channel), alpha is
        1 - d(fg) / d(plate), clamped to 0..1 (1 where the plate carries no screen). Optional
        `luminance_match` scales the plate toward the foreground's brightness (the background's when
        `use_bg_luminance` is on and bg is wired) by `luminance_level`. The output is premultiplied:
        with `screen_subtraction` the plate is subtracted from the foreground by the transparent
        share (1 - alpha), otherwise the foreground is scaled by alpha."""
        fg_rgb, plate_rgb = fg[..., :3], plate[..., :3]
        dominant = 2 if p.get("screen_type", "green") == "blue" else 1
        rw, bgw = float(p.get("red_weight", 0.5)), float(p.get("blue_green_weight", 0.5))
        # The weights apply to red and to the remaining (blue for green screens, green for blue).
        other = 2 if dominant == 1 else 1

        def difference(rgb):
            return rgb[..., dominant:dominant + 1] - (rw * rgb[..., 0:1] + bgw * rgb[..., other:other + 1])

        if p.get("luminance_match"):
            reference = bg[..., :3] if (p.get("use_bg_luminance") and bg is not None) else fg_rgb
            gain = np.clip(Evaluator._luma(reference) / np.maximum(Evaluator._luma(plate_rgb), 1e-4), 0.0, 4.0)
            plate_rgb = plate_rgb * (1.0 + float(p.get("luminance_level", 1.0)) * (gain - 1.0))
        plate_d = difference(plate_rgb)
        alpha = np.where(plate_d > 1e-4,
                         np.clip(1.0 - difference(fg_rgb) / np.maximum(plate_d, 1e-4), 0.0, 1.0), 1.0)
        if p.get("screen_subtraction", 1):
            rgb = np.maximum(fg_rgb - (1.0 - alpha) * plate_rgb, 0.0)
        else:
            rgb = fg_rgb * alpha
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _rgb_to_hue_sat(rgb):
        """Standard HSV hue (degrees, 0..360) and saturation (0..1) for a premultiplied-agnostic
        RGB triplet -- both are ratios of the raw channel values, so premult status doesn't matter
        the way it would for an additive quantity."""
        r, g, b = rgb[..., 0:1], rgb[..., 1:2], rgb[..., 2:3]
        mx = np.maximum(np.maximum(r, g), b)
        mn = np.minimum(np.minimum(r, g), b)
        delta = mx - mn
        safe_delta = np.where(delta > 1e-12, delta, 1.0)
        hue = np.where(mx == r, ((g - b) / safe_delta) % 6.0,
                       np.where(mx == g, ((b - r) / safe_delta) + 2.0, ((r - g) / safe_delta) + 4.0))
        hue = np.where(delta > 1e-12, hue * 60.0, 0.0) % 360.0
        safe_mx = np.where(mx > 1e-12, mx, 1.0)
        sat = np.where(mx > 1e-12, delta / safe_mx, 0.0)
        return hue, sat

    @staticmethod
    def _edge_falloff(distance, plateau, softness):
        """1 at or inside `plateau`, ramping down to 0 over the next `softness`, 0 beyond. A
        continuous formula rather than a branch on `softness == 0`: the tiny `1e-6` floor on the
        span makes that case an effectively-sharp (not literally infinite-slope) cutoff, which
        keeps this differentiable-everywhere like the rest of the evaluator's math."""
        span = max(float(softness), 1e-6)
        return np.clip((plateau + span - distance) / span, 0.0, 1.0)

    @staticmethod
    def _hue_keyer(image, p):
        rgb = image[..., :3]
        hue, sat = Evaluator._rgb_to_hue_sat(rgb)
        center = float(p.get("hue_center", 0.0)) % 360.0
        raw_dist = np.abs(hue - center) % 360.0
        dist = np.minimum(raw_dist, 360.0 - raw_dist)
        hue_gate = Evaluator._edge_falloff(dist, float(p.get("hue_width", 30.0)) / 2.0,
                                           p.get("hue_softness", 15.0))
        sat_gate = ((sat >= p.get("sat_min", 0.0)) & (sat <= p.get("sat_max", 1.0))).astype(np.float32)
        alpha = (hue_gate * sat_gate).astype(np.float32)
        if p.get("invert"):
            alpha = 1.0 - alpha
        return np.concatenate([rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _box_extreme_axis(frame, support, axis, use_max):
        """One separable pass of a box min/max filter: like `_box_blur_axis`'s sliding window, but
        min/max cannot use a cumulative-sum shortcut, so this walks the `2*support+1` offsets
        directly. "edge" padding matches Blur's own border handling."""
        if support <= 0:
            return frame
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (support, support)
        padded = np.pad(frame, pad, mode="edge")
        n = frame.shape[axis]
        reduce_fn = np.maximum if use_max else np.minimum
        result = None
        for offset in range(2 * support + 1):
            sl = [slice(None)] * frame.ndim
            sl[axis] = slice(offset, offset + n)
            window = padded[tuple(sl)]
            result = window if result is None else reduce_fn(result, window)
        return result

    @staticmethod
    def _box_extreme(image, size, use_max):
        # A box min/max filter is separable (unlike a general convolution) because the box
        # structuring element itself separates into independent x and y passes.
        support = 0 if abs(size) < 0.5 else int(math.ceil(abs(size)))
        if support == 0:
            return image.copy()
        return Evaluator._box_extreme_axis(
            Evaluator._box_extreme_axis(image, support, axis=1, use_max=use_max),
            support, axis=0, use_max=use_max)

    @staticmethod
    def _morph(image, size, channels):
        """Shared box erode/dilate kernel: positive `size` erodes (min filter, shrinks bright
        regions), negative `size` dilates (max filter, grows them) -- exactly Nuke's own signed
        Erode (fast) "size" knob."""
        out = image.copy()
        filtered = Evaluator._box_extreme(image, size, use_max=(size < 0))
        for c in Evaluator._CHANNEL_SETS[channels]:
            out[..., c] = filtered[..., c]
        return out

    @staticmethod
    def _erode(image, p):
        return Evaluator._morph(image, p.get("erode_size", 1.0), p.get("channels", "rgba"))

    @staticmethod
    def _dilate(image, p):
        # Dilate is Erode's positive twin: a positive dilate_size must grow, so it is handed to
        # the shared kernel negated (Erode's own convention for "grow").
        return Evaluator._morph(image, -p.get("dilate_size", 1.0), p.get("channels", "rgba"))

    @staticmethod
    def _median(image, p):
        size = p.get("median_size", 1.0)
        support = 0 if size < 0.5 else int(math.ceil(size))
        out = image.copy()
        if support == 0:
            return out
        h, w = image.shape[:2]
        padded = np.pad(image, ((support, support), (support, support), (0, 0)), mode="edge")
        window = 2 * support + 1
        # A true 2D median is not separable, unlike box blur/erode/dilate, so every tap in the
        # window is stacked and reduced at once. Fine at the small radii these tests and typical
        # despeckle work use; a large radius would want a running-histogram median instead.
        stack = np.stack([padded[dy:dy + h, dx:dx + w] for dy in range(window) for dx in range(window)])
        med = np.median(stack, axis=0).astype(np.float32)
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c] = med[..., c]
        return out

    @staticmethod
    def _matrix(image, p, laplacian=False):
        """Apply the selected odd square convolution; edge pixels extend nearest samples."""
        if laplacian:
            weights = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
            normalize = False
        else:
            size = int(str(p.get("matrix_size", "3")) or 3)
            size = size if size in (3, 5, 7) else 3
            weights = np.array([float(p.get(f"weight{i}", 0.0)) for i in range(size * size)], dtype=np.float32).reshape(size, size)
            normalize = bool(p.get("normalize", 0))
        if normalize:
            total = float(weights.sum())
            if abs(total) > 1e-12:
                weights /= total
        identity = np.zeros_like(weights)
        identity[weights.shape[0] // 2, weights.shape[1] // 2] = 1.0
        if np.array_equal(weights, identity):
            return image.copy()
        h, w = image.shape[:2]
        radius = weights.shape[0] // 2
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        out = np.zeros_like(image)
        for y in range(weights.shape[0]):
            for x in range(weights.shape[1]):
                out += padded[y:y+h, x:x+w] * weights[y, x]
        result = image.copy()
        for c in Evaluator._CHANNEL_SETS.get(p.get("channels", "rgba"), (0, 1, 2, 3)):
            result[..., c] = out[..., c]
        return result.astype(np.float32)

    @staticmethod
    def _convolve(image, kernel, p):
        size = int(str(p.get("kernel_size", "1")) or 1)
        size = size if size in (1, 3, 5, 7) else 1
        if kernel.shape[0] < size or kernel.shape[1] < size:
            raise ValueError(f"Convolve kernel image must be at least {size} by {size}")
        weights = np.array(kernel[:size, :size, 0], dtype=np.float32, copy=True)
        total = float(weights.sum())
        if abs(total) < 1e-12:
            raise ValueError("Convolve kernel image must have a non-zero sum")
        weights /= total
        h, w = image.shape[:2]; radius = size // 2
        padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        out = np.zeros_like(image)
        for y in range(size):
            for x in range(size): out += padded[y:y+h, x:x+w] * weights[y, x]
        return out.astype(np.float32)

    @staticmethod
    def _edge_detect(image, p):
        rgb = image[..., :3].mean(axis=2)
        method = p.get("edge_type", "Sobel")
        if method == "Laplacian":
            response = np.abs(Evaluator._matrix(np.repeat(rgb[..., None], 4, axis=2), {}, laplacian=True)[..., 0])
        else:
            a = np.pad(rgb, 1, mode="edge")
            if method == "Prewitt":
                gx = (a[:-2, 2:] + a[1:-1, 2:] + a[2:, 2:] - a[:-2, :-2] - a[1:-1, :-2] - a[2:, :-2]) / 3.0
                gy = (a[2:, :-2] + a[2:, 1:-1] + a[2:, 2:] - a[:-2, :-2] - a[:-2, 1:-1] - a[:-2, 2:]) / 3.0
            else:
                gx = a[:-2, 2:] + 2*a[1:-1, 2:] + a[2:, 2:] - a[:-2, :-2] - 2*a[1:-1, :-2] - a[2:, :-2]
                gy = a[2:, :-2] + 2*a[2:, 1:-1] + a[2:, 2:] - a[:-2, :-2] - 2*a[:-2, 1:-1] - a[:-2, 2:]
            response = np.hypot(gx, gy).astype(np.float32)
        response = np.maximum(response - float(p.get("threshold", 0.0)), 0.0)
        out = image.copy()
        out[..., :3] = response[..., None]
        return out

    @staticmethod
    def _emboss(image, p):
        lum = image[..., :3].mean(axis=2)
        a = np.pad(lum, 1, mode="edge")
        angle = math.radians(float(p.get("angle", 135.0)))
        dx = (a[1:-1, 2:] - a[1:-1, :-2]) * 0.5
        dy = (a[2:, 1:-1] - a[:-2, 1:-1]) * 0.5
        shade = np.clip(0.5 + float(p.get("width", 1.0)) * (np.cos(angle)*dx + np.sin(angle)*dy), 0, 1)
        out = image.copy(); out[..., :3] = shade[..., None]
        return out

    @staticmethod
    def _bump_boss(image, p):
        channel = p.get("height_channel", "rgba.red").split(".")[-1]
        index = {"red": 0, "green": 1, "blue": 2, "alpha": 3}.get(channel, 0)
        h = image[..., index]; a = np.pad(h, 1, mode="edge")
        dx = (a[1:-1, 2:] - a[1:-1, :-2]) * 0.5
        dy = (a[2:, 1:-1] - a[:-2, 1:-1]) * 0.5
        angle = math.radians(float(p.get("light_angle", 135.0)))
        shade = np.clip(0.5 + np.cos(angle)*dx + np.sin(angle)*dy, 0, 1)
        out = image.copy(); out[..., :3] = shade[..., None]
        return out

    @staticmethod
    def _contact_sheet_cell(index, cols, rows, roworder, colorder, center=False, count=None):
        """The (row, col) a clip lands in, matching Nuke's `roworder`/`colorder` knobs.

        Clips fill logical rows top-to-bottom, left-to-right first; `colorder` then reverses a
        row (RightLeft) or alternates direction per row (Snake), and `roworder` maps that logical
        row onto the canvas from the top (TopBottom) or, Nuke's default, from the bottom
        (BottomTop) so the first clips land in the bottom row.

        `center` (step D2 finish), with the total placed `count`, pads a grid the clips do not
        completely fill instead of anchoring the block at roworder/colorder's own starting corner:
        empty logical rows split before the used block (vertical centring), and the last, possibly
        partial, logical row is padded on both sides (horizontal centring of that row only, the
        usual contact-sheet convention). A fully filled grid (`count == rows * cols`) pads by zero.
        """
        logical_row, col = divmod(index, cols)
        if center and count:
            used_rows = -(-count // cols)   # ceil(count / cols)
            last_row_count = count - (used_rows - 1) * cols
            logical_row += (rows - used_rows) // 2
            if index >= (used_rows - 1) * cols:   # the last (possibly partial) logical row
                col += (cols - last_row_count) // 2
        if colorder == "RightLeft" or (colorder == "Snake" and logical_row % 2 == 1):
            col = cols - 1 - col
        row = logical_row if roworder == "TopBottom" else rows - 1 - logical_row
        return row, col

    @staticmethod
    def _erode_filter(image, p):
        # Distance-falloff erode: blend the hard erosions at the surrounding integer radii by how
        # far `size` sits between them, shaped by the selected filter kernel. Foundry does not
        # publish Erode (filter)'s exact analytic response, so the falloff curves are this
        # repository's own choice (unverified against Nuke), picked to match the documented
        # box < triangle < quadratic < gaussian smoothness ordering: box ramps linearly, triangle
        # and quadratic are progressively smoother polynomial ease curves, and gaussian instead
        # reaches for a true post-erode blur, as it already did.
        size = float(p.get("filter_size", 1.0))
        if size < 0.5:
            return image.copy()
        filter_type = p.get("filter_type", "box")

        def erode_at(radius):
            if radius <= 0:
                return image.copy()
            h, w = image.shape[:2]
            padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
            windows = [padded[radius+dy:radius+dy+h, radius+dx:radius+dx+w]
                       for dy in range(-radius, radius+1) for dx in range(-radius, radius+1)
                       if dx*dx + dy*dy <= radius*radius + 1e-9]
            return np.minimum.reduce(windows).astype(np.float32)

        r0 = int(math.floor(size))
        frac = size - r0
        hard = erode_at(r0)
        if frac > 1e-9:
            hard1 = erode_at(r0 + 1)
            if filter_type == "triangle":
                w = frac * frac * (3.0 - 2.0 * frac)                          # cubic smoothstep, C1
            elif filter_type == "quadratic":
                w = frac * frac * frac * (frac * (frac * 6.0 - 15.0) + 10.0)  # quintic ease, C2
            else:
                w = frac                                                      # box: linear ramp
            hard = hard * (1.0 - w) + hard1 * w
        if filter_type == "gaussian":
            hard = Evaluator._soften(hard, {"soften_size": max(1.0, size * 0.5)})
        return hard.astype(np.float32)

    @staticmethod
    def _sharpen(image, p):
        # Unsharp mask: add back `amount` of the high-frequency detail a box blur removed.
        # A flat image has no detail (blurred == image), so it is the identity at any amount.
        amount = p.get("sharpen_amount", 0.5)
        size = p.get("sharpen_size", 1.0)
        blurred = Evaluator._blur(image, {"radius": size})
        detail = image - blurred
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = image[..., c] + detail[..., c] * amount
        return out

    @staticmethod
    def _glow(image, p):
        # Nuke's Glow2: blur only what's above a threshold, tint and scale it, and add it back
        # over the input. A black image has nothing above a positive threshold, so it stays black.
        threshold = p.get("glow_threshold", 1.0)
        size = p.get("glow_size", 8.0)
        bright = np.clip(image[..., :3] - threshold, 0.0, None)
        blurred = Evaluator._blur(np.concatenate([bright, image[..., 3:4]], axis=2),
                                  {"radius": size})[..., :3]
        brightness = p.get("brightness", 1.0)
        tint = np.array([p.get("red", 1.0), p.get("green", 1.0), p.get("blue", 1.0)], dtype=np.float32)
        glow_rgb = (blurred * brightness * tint).astype(np.float32)
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            if c < 3:
                out[..., c] = image[..., c] + glow_rgb[..., c]
        return out

    @staticmethod
    def _exposure(image, p):
        # Nuke's Exposure: out = (in - blackpoint) * gain, per channel, so a pixel sitting at the
        # black point becomes exactly 0 and everything else scales about it. `stops` gain is
        # 2 ** exposure (the same stop math as Grade.exposure); `densities` gain is
        # 10 ** (density / 0.6), the reference guide's "log10(density) of 0.6 gamma negative
        # stock". With `gang` on the red slider drives all three channels. Alpha (channels =
        # rgba/alpha) takes the red channel's gain.
        base = 2.0 if p.get("exposure_mode", "stops") == "stops" else 10.0
        scale = 1.0 if base == 2.0 else 1.0 / 0.6
        red = float(p.get("red", 0.0))
        if p.get("gang", 1):
            amounts = (red, red, red)
        else:
            amounts = (red, float(p.get("green", 0.0)), float(p.get("blue", 0.0)))
        gains = [np.float32(base ** (a * scale)) for a in amounts]
        gains.append(gains[0])
        black = np.float32(p.get("blackpoint", 0.0))
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = (image[..., c] - black) * gains[c]
        return out

    @staticmethod
    def _hue_correct(image, p):
        from . import colorcurves
        rgb = image[..., :3]
        hue, sat = Evaluator._rgb_to_hue_sat(rgb)
        hue = hue[..., 0].astype(np.float64) % 360.0
        sat = sat[..., 0]
        default_curve = '{"interpolation":"linear","points":[[0,1],[360,1]]}'
        legacy_names = ("red", "yellow", "green", "cyan", "blue", "magenta")
        for prefix in ("sat", "lum"):
            values = [float(p.get(f"{prefix}_{band}", 1.0)) for band in legacy_names]
            raw = p.get(f"curve_{prefix}", default_curve)
            if raw == default_curve and any(value != 1.0 for value in values):
                from .colorcurves import anchor_curve
                p = dict(p, **{f"curve_{prefix}": anchor_curve(values)})
        def sample(name, default=1.0):
            raw = p.get(f"curve_{name}")
            if raw is None:
                return np.full(hue.shape, default, np.float32)
            return colorcurves.evaluate_array(colorcurves.decode(raw), hue)
        s_mult, l_mult = sample("sat"), sample("lum")
        luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
        out = rgb + (s_mult[..., None] - 1.0) * (rgb - luma)
        out = out * (1.0 + (l_mult[..., None] - 1.0) * np.clip(sat[..., None], 0.0, 1.0))
        for channel, name in enumerate(("red", "green", "blue")):
            out[..., channel] *= sample(name)
        # Suppression is despill-style limiting, a separate output from the channel response above:
        # a channel is pulled toward the larger of the other two by the curve's value, so 1 limits
        # it to them and a channel already below them is left alone.
        pre = out.copy()
        for channel, name in enumerate("rgb"):
            others = np.maximum(pre[..., (channel + 1) % 3], pre[..., (channel + 2) % 3])
            out[..., channel] -= sample(name + "_sup", 0.0) * np.maximum(pre[..., channel] - others, 0.0)
        shift = float(p.get("hue_shift", 0.0))
        if shift % 360.0 != 0.0:
            a = math.radians(shift); c, s = math.cos(a), math.sin(a); k = 1.0 / math.sqrt(3.0)
            m = np.array([[c + (1-c)/3, (1-c)/3-s*k, (1-c)/3+s*k],
                          [(1-c)/3+s*k, c + (1-c)/3, (1-c)/3-s*k],
                          [(1-c)/3-s*k, (1-c)/3+s*k, c + (1-c)/3]], np.float32)
            out = out @ m.T
        return np.concatenate([out, image[..., 3:4]], axis=2).astype(np.float32)
    @staticmethod
    def _color_lookup(image, p):
        from . import colorcurves
        result = image.copy()
        master = colorcurves.decode(p["curve_master"])
        for index, name in enumerate(("red", "green", "blue", "alpha")):
            values = colorcurves.evaluate_array(master, result[..., index])
            result[..., index] = colorcurves.evaluate_array(colorcurves.decode(p[f"curve_{name}"]), values)
        return result

    @staticmethod
    def _color_matrix(image, p):
        # ColorMatrix: out = M @ (r, g, b) with M read row by row from matrix_RC. With `invert` on
        # the inverse is applied instead; a singular matrix (|det| < 1e-9) has no inverse, so the
        # node then passes the input through unchanged rather than emitting NaNs. Alpha is untouched.
        m = np.array([[float(p.get(f"matrix_{i}{j}", 1.0 if i == j else 0.0)) for j in range(3)]
                      for i in range(3)], np.float64)
        if p.get("invert"):
            if abs(np.linalg.det(m)) < 1e-9:
                return image.copy()
            m = np.linalg.inv(m)
        m = m.astype(np.float32)
        rgb = image[..., :3]
        out = np.stack([m[i, 0] * rgb[..., 0] + m[i, 1] * rgb[..., 1] + m[i, 2] * rgb[..., 2]
                        for i in range(3)], axis=2)
        return np.concatenate([out, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _gaussian_axis(frame, radius, sigma, axis, mode="edge"):
        """Separable Gaussian pass with a truncated, renormalised kernel of half-width `radius`
        (weights sum to exactly 1, so flat regions and total energy are preserved) and "edge"
        padding like `_box_blur_axis`."""
        offsets = np.arange(-radius, radius + 1, dtype=np.float64)
        weights = np.exp(-0.5 * (offsets / sigma) ** 2)
        weights /= weights.sum()
        n = frame.shape[axis]
        pad = [(0, 0)] * frame.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(frame, pad, mode=mode).astype(np.float64)
        out = np.zeros(frame.shape, dtype=np.float64)
        for i, weight in enumerate(weights):
            index = [slice(None)] * frame.ndim
            index[axis] = slice(i, i + n)
            out += weight * padded[tuple(index)]
        return out.astype(np.float32)

    @staticmethod
    def _soften(image, p):
        # Nuke's Soften: a Gaussian-leaning blur. `soften_size` is the kernel's pixel reach:
        # sigma = size / 3, truncated at ceil(size) pixels (three sigma), which is exactly the
        # padded support `tiers._support_rule` declares, so tiles have every neighbour they read.
        # Below the same 0.5 cut-off the other padded filters use, it is the identity.
        size = abs(float(p.get("soften_size", 4.0)))
        if size < 0.5:
            return image.copy()
        radius = int(math.ceil(size))
        sigma = size / 3.0
        soft = Evaluator._gaussian_axis(Evaluator._gaussian_axis(image, radius, sigma, axis=1),
                                        radius, sigma, axis=0)
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c] = soft[..., c]
        return out

    @staticmethod
    def _disc_radii(p):
        """Semi-axes (rx, ry) of Defocus's disc: `defocus` wide, `aspect` = width / height."""
        radius = abs(float(p.get("defocus", 6.0)))
        aspect = float(p.get("aspect", 1.0))
        return radius, radius / aspect if aspect > 0 else radius

    @staticmethod
    def _defocus(image, p):
        # Nuke's Defocus without depth: every output pixel is the plain average of the input pixels
        # whose centres fall inside an ellipse of semi-axes (defocus, defocus / aspect), so a
        # bright point becomes a flat-topped disc (unlike Blur's box and Soften's Gaussian) and
        # the weights sum to exactly 1. The disc is summed one row at a time: row dy contributes a
        # horizontal box of half-width floor(rx * sqrt(1 - (dy / ry) ** 2)), built from a running
        # sum, so cost grows with the radius, not its square. "edge" padding, like Blur. Below a
        # half pixel it is the identity, the cut-off the other padded filters use.
        rx, ry = Evaluator._disc_radii(p)
        if max(rx, ry) < 0.5:
            return image.copy()
        rows = int(math.floor(ry + 1e-9))
        half_widths = {}
        for dy in range(-rows, rows + 1):
            inside = max(0.0, 1.0 - (dy / ry) ** 2)
            half_widths[dy] = int(math.floor(rx * math.sqrt(inside) + 1e-9))
        reach = max(half_widths.values())
        total = float(sum(2 * w + 1 for w in half_widths.values()))
        height, width = image.shape[:2]
        padded = np.pad(image, ((rows, rows), (reach, reach), (0, 0)), mode="edge").astype(np.float64)
        running = np.concatenate([np.zeros((padded.shape[0], 1, 4)), np.cumsum(padded, axis=1)], axis=1)
        out = np.zeros(image.shape, dtype=np.float64)
        for dy, w in half_widths.items():
            band = running[rows + dy:rows + dy + height]
            out += band[:, reach + w + 1:reach + w + 1 + width] - band[:, reach - w:reach - w + width]
        out /= total
        result = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            result[..., c] = out[..., c].astype(np.float32)
        return result

    @staticmethod
    def _bilateral(image, p):
        """Small bilateral filter; edge-aware RGB weights, spatial Gaussian and extended borders."""
        radius = int(math.ceil(max(0.0, float(p.get("spatial_size", 3.0)))))
        if radius == 0:
            return image.copy()
        sigma = max(1e-5, float(p.get("colour_sigma", 0.1)))
        h, w = image.shape[:2]
        pad = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        out = np.zeros_like(image, dtype=np.float64)
        total = np.zeros((h, w, 1), dtype=np.float64)
        center = image.astype(np.float64)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                spatial = math.exp(-(dx * dx + dy * dy) / max(1.0, 2.0 * radius * radius))
                sample = pad[radius + dy:radius + dy + h, radius + dx:radius + dx + w].astype(np.float64)
                diff = sample[..., :3] - center[..., :3]
                weight = spatial * np.exp(-np.sum(diff * diff, axis=2, keepdims=True) / (2 * sigma * sigma))
                out += sample * weight
                total += weight
        return (out / np.maximum(total, 1e-20)).astype(np.float32)

    @staticmethod
    def _degrain_simple(image, p):
        result = image.copy()
        for channel, key in enumerate(("red_amount", "green_amount", "blue_amount")):
            amount = float(p.get(key, 0.0))
            if amount > 0.05:
                result[..., channel:channel + 1] = Evaluator._blur(image[..., channel:channel + 1].repeat(4, axis=2), {"radius": amount})[..., :1]
        return result

    @staticmethod
    def _zdefocus(image, depth, p, kernel=None):
        """Depth-radius disc blur, composited nearest-to-farthest to protect foreground edges."""
        if depth is None:
            raise ValueError("ZDefocus needs a depth layer on the image or a wired depth input")
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3: depth = depth[..., 0]
        if depth.shape != image.shape[:2]:
            raise ValueError("ZDefocus depth must match the image dimensions")
        z = np.maximum(depth, 1e-6)
        if p.get("depth_math") == "1/depth": z = 1.0 / z
        focal = float(p.get("focal_plane", 1.0)); dof = max(1e-6, float(p.get("depth_of_field", 1.0)))
        radii = np.minimum(float(p.get("max_size", 20.0)), np.abs(z - focal) / dof * float(p.get("max_size", 20.0)))
        radius = int(math.ceil(float(np.max(radii)))) if radii.size else 0
        if radius <= 0: return image.copy()
        h, w = depth.shape
        pad_img = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode="edge")
        pad_z = np.pad(z, radius, mode="edge")
        out = np.empty_like(image)
        yy, xx = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        for y in range(h):
            for x in range(w):
                r = float(radii[y, x])
                if p.get("bokeh_shape") == "image":
                    if kernel is None:
                        raise ValueError("ZDefocus image bokeh requires a wired kernel input")
                    source_kernel = np.asarray(kernel, dtype=np.float32)
                    if source_kernel.ndim == 3: source_kernel = source_kernel[..., 3] if source_kernel.shape[2] > 3 else source_kernel[..., 0]
                    kh, kw = source_kernel.shape[:2]
                    ky = np.minimum(kh - 1, np.maximum(0, ((np.arange(2 * radius + 1) - radius + radius) / max(1, 2 * radius) * (kh - 1)).astype(int)))
                    kx = np.minimum(kw - 1, np.maximum(0, ((np.arange(2 * radius + 1) - radius + radius) / max(1, 2 * radius) * (kw - 1)).astype(int)))
                    kernel_grid = source_kernel[np.ix_(ky, kx)]
                    disk = kernel_grid > 0
                elif p.get("bokeh_shape") == "blades" and r > 0:
                    theta = np.arctan2(yy, xx) - math.radians(float(p.get("blade_rotation", 0.0)))
                    sector = (theta + math.pi / int(p.get("blade_count", 6))) % (2 * math.pi / int(p.get("blade_count", 6))) - math.pi / int(p.get("blade_count", 6))
                    boundary = r * math.cos(math.pi / int(p.get("blade_count", 6))) / np.maximum(np.cos(sector), 1e-6)
                    disk = xx * xx + yy * yy <= boundary * boundary
                else:
                    disk = xx * xx + yy * yy <= r * r
                samples_z = pad_z[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
                # Near objects remain in front; reject farther samples around a foreground pixel.
                visible = disk & (samples_z <= z[y, x] + 1e-6)
                if not np.any(visible): visible[radius, radius] = True
                samples = pad_img[y:y + 2 * radius + 1, x:x + 2 * radius + 1]
                if p.get("bokeh_shape") == "image" and kernel is not None:
                    weights = kernel_grid * visible
                    out[y, x] = np.sum(samples * weights[..., None], axis=(0, 1)) / max(float(weights.sum()), 1e-20)
                else:
                    out[y, x] = samples[visible].mean(axis=0)
        return out

    @staticmethod
    def _shadow_offset(p):
        """Whole-pixel (dx, dy) of DropShadow's offset, in array coordinates (y down)."""
        angle = math.radians(float(p.get("angle", -45.0)))
        distance = abs(float(p.get("distance", 0.0)))
        return int(round(distance * math.cos(angle))), int(round(-distance * math.sin(angle)))

    @staticmethod
    def _drop_shadow(image, p):
        # Nuke's DropShadow: the input's alpha is moved by (distance, angle), rounded to whole
        # pixels so the shadow lands exactly where the offset says, blurred by `shadow_size` (the
        # Soften kernel: sigma = size / 3, truncated at ceil(size) pixels), scaled by `opacity`,
        # tinted, and composited UNDER the input: out = input + shadow * (1 - input alpha) on
        # premultiplied RGBA, so wherever the input is opaque it is untouched. Outside the frame
        # counts as transparent (zero fill, not edge extension: a shadow must not be invented from
        # the border pixel). The result stays inside the input's window, like Blur.
        height, width = image.shape[:2]
        dx, dy = Evaluator._shadow_offset(p)
        shifted = np.zeros((height, width, 1), dtype=np.float32)
        src_x, src_y = slice(max(0, -dx), min(width, width - dx)), slice(max(0, -dy), min(height, height - dy))
        dst_x, dst_y = slice(max(0, dx), min(width, width + dx)), slice(max(0, dy), min(height, height + dy))
        if dst_x.stop > dst_x.start and dst_y.stop > dst_y.start:
            shifted[dst_y, dst_x, 0] = image[src_y, src_x, 3]
        size = abs(float(p.get("shadow_size", 0.0)))
        if size >= 0.5:
            radius, sigma = int(math.ceil(size)), size / 3.0
            shifted = Evaluator._gaussian_axis(Evaluator._gaussian_axis(shifted, radius, sigma, axis=1, mode="constant"),
                                               radius, sigma, axis=0, mode="constant")
        shadow = shifted * np.float32(p.get("opacity", 0.5))
        tint = np.array([p.get("red", 0.0), p.get("green", 0.0), p.get("blue", 0.0), 1.0], dtype=np.float32)
        return (image + shadow * tint * (1.0 - image[..., 3:4])).astype(np.float32)

    @staticmethod
    def _gaussian_2d(frame, size):
        """Soften's Gaussian (sigma = size / 3, truncated at ceil(size) pixels, "edge" padding);
        the identity below the half-pixel cut-off the padded filters share."""
        size = abs(float(size))
        if size < 0.5:
            return frame
        radius, sigma = int(math.ceil(size)), size / 3.0
        return Evaluator._gaussian_axis(Evaluator._gaussian_axis(frame, radius, sigma, axis=1),
                                        radius, sigma, axis=0)

    @staticmethod
    def _reach(size):
        """Pixels a `_gaussian_2d` / `_box_extreme` of `size` reads each way."""
        size = abs(float(size))
        return 0 if size < 0.5 else int(math.ceil(size))

    @staticmethod
    def _edge_blur(image, p):
        # Nuke's EdgeBlur: blur only where the matte has an edge. The band is the alpha's own
        # morphological gradient, box dilate minus box erode over `edge_mult * size` pixels, so
        # it is 1 for a hard matte within that distance of the edge on both sides, fractional
        # across a soft edge, and 0 in a flat interior or exterior. The result is
        # image + band * (Gaussian(image, size) - image): pixels outside the band are the input,
        # bit for bit, even when they hold detail a plain Blur would smear.
        size = abs(float(p.get("edgeblur_size", 4.0)))
        out = image.copy()
        band_width = size * max(0.0, float(p.get("edge_mult", 1.0)))
        if size < 0.5 or band_width < 0.5:
            return out
        alpha = image[..., 3:4]
        band = np.clip(Evaluator._box_extreme(alpha, band_width, use_max=True)
                       - Evaluator._box_extreme(alpha, band_width, use_max=False), 0.0, 1.0)
        blurred = Evaluator._gaussian_2d(image, size)
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c] = image[..., c] + band[..., 0] * (blurred[..., c] - image[..., c])
        return out.astype(np.float32)

    @staticmethod
    def _edge_extend(image, p):
        # Pushes edge colour outward past the matte to kill dark fringes. The input is
        # premultiplied; the output is its UNPREMULTIPLIED colour (follow with Premult, or feed a
        # filter that would otherwise bleed black), alpha untouched. Pixels with alpha at or above
        # `extend_threshold` are trusted sources. Every other pixel, including the low-alpha edge
        # pixels that carry the fringe, takes the mean colour of its already-known 8 neighbours,
        # one ring per step, for ceil(extend_size) steps: colour reaches ceil(extend_size) pixels
        # out, which is the padded support `tiers._edge_extend_rule` declares. Pixels no source
        # reaches keep their own straight colour (zero where alpha is zero).
        alpha = image[..., 3]
        threshold = max(float(p.get("extend_threshold", 0.5)), 1e-6)
        safe = np.where(alpha > 0, alpha, 1.0)[..., None]
        straight = np.where(alpha[..., None] > 0, image[..., :3] / safe, 0.0).astype(np.float32)
        known = alpha >= threshold
        colour = np.where(known[..., None], straight, 0.0).astype(np.float32)
        height, width = alpha.shape
        for _ in range(Evaluator._reach(p.get("extend_size", 3.0))):
            weight = np.pad(known.astype(np.float32), 1)
            padded = np.pad(colour * known[..., None], ((1, 1), (1, 1), (0, 0)))
            total = np.zeros_like(colour)
            count = np.zeros(alpha.shape, dtype=np.float32)
            for dy in (0, 1, 2):
                for dx in (0, 1, 2):
                    if dy == 1 and dx == 1:
                        continue
                    total += padded[dy:dy + height, dx:dx + width]
                    count += weight[dy:dy + height, dx:dx + width]
            fresh = (~known) & (count > 0)
            colour = np.where(fresh[..., None], total / np.maximum(count, 1.0)[..., None], colour)
            known = known | fresh
        rgb = np.where(known[..., None], colour, straight)
        return np.concatenate([rgb, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _light_reach(p):
        """Pixels `_light_wrap` reads each way: the foreground alpha is blurred by `fgblur` and
        then its inverse by `wrap_diffuse` (the two reaches add); the background by `bgblur`."""
        return max(Evaluator._reach(p.get("fgblur", 1.0)) + Evaluator._reach(p.get("wrap_diffuse", 10.0)),
                   Evaluator._reach(p.get("bgblur", 4.0)))

    @staticmethod
    def _light_wrap(fg, bg, p):
        # Nuke's LightWrap: light from the background spills around the foreground's edge. The
        # foreground alpha is softened by `fgblur`; its inverse is blurred by `wrap_diffuse` and
        # doubled (a straight edge reads 0.5 there, so the doubling gives full strength exactly
        # at the edge), then multiplied by the foreground alpha so nothing is added outside the
        # matte and nothing deep inside it. The wrapped light is the background blurred by
        # `bgblur`, minus `wrap_threshold` (floored at 0), or a constant colour when
        # `use_constant_highlight` is on. With e = edge * intensity, `highlight_merge` folds it
        # into the premultiplied foreground: plus fg + light * e; screen adds it as 1 - (1 - fg)
        # * (1 - light * e); max takes max(fg, light * e); over is light * e over fg. Alpha is
        # the foreground's, unchanged.
        alpha = fg[..., 3:4]
        soft_alpha = Evaluator._gaussian_2d(alpha, p.get("fgblur", 1.0))
        if Evaluator._reach(p.get("wrap_diffuse", 10.0)) == 0:
            edge = np.zeros_like(alpha)
        else:
            edge = alpha * np.clip(2.0 * Evaluator._gaussian_2d(1.0 - soft_alpha, p.get("wrap_diffuse", 10.0)), 0.0, 1.0)
        if p.get("use_constant_highlight", 0):
            light = np.broadcast_to(np.array([p.get("red", 1.0), p.get("green", 1.0), p.get("blue", 1.0)],
                                             dtype=np.float32), fg[..., :3].shape)
        else:
            light = np.clip(Evaluator._gaussian_2d(bg[..., :3], p.get("bgblur", 4.0))
                            - np.float32(p.get("wrap_threshold", 0.0)), 0.0, None)
        wrap = light * edge * np.float32(p.get("intensity", 1.0))
        base = fg[..., :3]
        operation = p.get("highlight_merge", "plus")
        if operation == "screen":
            rgb = base + wrap * (1.0 - base)
        elif operation == "max":
            rgb = np.maximum(base, wrap)
        elif operation == "over":
            rgb = wrap + base * (1.0 - np.clip(edge * np.float32(p.get("intensity", 1.0)), 0.0, 1.0))
        else:
            rgb = base + wrap
        return np.concatenate([rgb, fg[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _dither(image, p, origin=(0, 0)):
        # Quantises to `bits` per channel with triangular (TPDF) noise of +/- `dither_amount`
        # least-significant bits added first, so a smooth gradient reduces to fine grain whose
        # average is the original value instead of contour bands. The noise is a hash of the
        # absolute pixel position, the channel and `seed`, never a random stream, so the same
        # seed gives identical pixels on every run, on the whole frame or one tile at a time
        # (`origin` is the array's canvas position; proxy tiers reach different noise, as they
        # reach different pixels). Values are clamped to 0..1 by the quantiser. Alpha is left
        # alone unless `channels` says otherwise.
        bits = int(min(16, max(1, round(float(p.get("bits", 8))))))
        levels = float(2 ** bits - 1)
        amount = float(p.get("dither_amount", 1.0))
        seed = int(p.get("seed", 0))
        height, width = image.shape[:2]
        iy, ix = np.mgrid[0:height, 0:width]
        ix, iy = ix + int(origin[0]), iy + int(origin[1])
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            noise = (Evaluator._hash_lattice(ix, iy, c, seed)
                     + Evaluator._hash_lattice(ix, iy, c, seed + 7919) - 1.0)
            out[..., c] = (np.clip(np.round(image[..., c] * levels + noise * amount), 0.0, levels)
                           / levels).astype(np.float32)
        return out

    @staticmethod
    def _premultiply(image):
        out = image.copy()
        out[..., :3] = image[..., :3] * image[..., 3:4]
        return out

    @staticmethod
    def _grain(image, p, origin=(0, 0), frame=None):
        # Synthetic grain: zero-mean lattice noise added per channel, so the mean of a flat area
        # is preserved. Each channel has its own size (the lattice spacing in pixels; 1 or less is
        # one independent value per pixel) and intensity (the amplitude of unit-variance noise at
        # size 1; larger sizes interpolate, which lowers the variance a little). The pattern is a
        # pure function of the absolute pixel position, the channel and `seed + frame`, so it is
        # the same on the whole frame and on any tile, moves every frame, and freezes when the
        # seed is set to minus the frame (Nuke's "-frame" recipe). `luminance_weighted` scales the
        # grain by clamp(luma, 0, 1), with `black` as the floor of that weight.
        height, width = image.shape[:2]
        xs = np.arange(width, dtype=np.float64) + int(origin[0])
        ys = np.arange(height, dtype=np.float64) + int(origin[1])
        gx, gy = np.meshgrid(xs, ys)
        seed = int(p.get("seed", 0)) + int(frame or 0)
        weight = 1.0
        if p.get("luminance_weighted"):
            rgb = image[..., :3]
            luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
            black = float(p.get("black", 0.0))
            weight = black + (1.0 - black) * np.clip(luma, 0.0, 1.0)
        out = image.copy()
        for c, name in enumerate(("red", "green", "blue")):
            intensity = float(p.get(f"{name}_intensity", 0.0))
            if intensity == 0.0:
                continue
            size = max(float(p.get(f"{name}_size", 1.0)), 1.0)
            noise = (Evaluator._value_noise(gx / size, gy / size, c, seed) - 0.5) * math.sqrt(12.0)
            out[..., c] = (image[..., c] + noise * intensity * weight).astype(np.float32)
        return out

    @staticmethod
    def _flare(image, p, origin=(0, 0), frame_size=None):
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        x, y = xx + origin[0], yy + origin[1]
        cx, cy = float(p.get("position_x", w / 2)), float(p.get("position_y", h / 2))
        dx, dy = x-cx, y-cy
        r = np.sqrt(dx*dx + dy*dy)
        size = max(0.1, float(p.get("size", 18)))
        glow = np.exp(-(r / size) ** 2)
        # A pair of low-amplitude diffraction rings around the bright core.
        glow += 0.12 * np.exp(-np.abs(r - size * 2.0) / max(0.4, size * 0.06))
        glow += 0.06 * np.exp(-np.abs(r - size * 3.2) / max(0.4, size * 0.05))
        angle = math.radians(float(p.get("rotation", 0)))
        streaks = max(0, min(64, int(p.get("streaks", 6))))
        length = max(0.0, float(p.get("length", 180)))
        for i in range(streaks):
            a = angle + i * math.pi / max(1, streaks)
            along = np.abs(dx*math.cos(a) + dy*math.sin(a))
            across = np.abs(-dx*math.sin(a) + dy*math.cos(a))
            glow += np.exp(-along / max(1, length)) * np.exp(-across / max(0.5, size*0.12)) * 0.18
        ghosts = max(0, min(64, int(p.get("ghosts", 5))))
        # Optical ghosts lie on the axis through the flare and the frame centre.
        frame_width, frame_height = frame_size or (w, h)
        vx, vy = frame_width / 2 - cx, frame_height / 2 - cy
        norm = max(1.0, math.hypot(vx, vy))
        for i in range(1, ghosts + 1):
            t = -i * float(p.get("spread", 0.65))
            gx, gy = cx + vx*t, cy + vy*t
            gr = np.sqrt((x-gx)**2 + (y-gy)**2)
            gs = size * max(0.1, 1.0 - i/(ghosts+1))
            glow += 0.25 * np.exp(-(gr/gs)**2)
        shift = float(p.get("chromatic_shift", 2.0))
        color = np.array([p.get("red", .7), p.get("green", .85), p.get("blue", 1.)], np.float32)
        out = image.copy()
        gain = float(p.get("brightness", 1.0))
        for c in range(3):
            delta = shift * (c-1)
            channel_r = np.sqrt((x-(cx+delta))**2 + (y-cy)**2)
            out[..., c] += (gain * color[c] * (glow + .5*np.exp(-(channel_r/size)**2))).astype(np.float32)
        return out

    @staticmethod
    def _glint(image, p):
        # A line-kernel convolution of only the values above tolerance. Every chosen orientation
        # contributes two symmetric arms, so `rays` is the count of distinct star axes.
        bright = np.maximum(image[..., :3] - float(p.get("tolerance", 1)), 0.0)
        out = image.copy()
        count = max(1, min(32, int(p.get("rays", 4))))
        length = max(0.0, float(p.get("length", 12)))
        h, w = bright.shape[:2]
        glint = np.zeros_like(bright)
        falloff = max(0.0, float(p.get("falloff", .8)))
        for i in range(count):
            a = math.radians(float(p.get("rotation", 0))) + i*math.pi/count
            dx, dy = int(round(math.cos(a))), int(round(math.sin(a)))
            if dx == 0 and dy == 0: dx = 1
            for distance in range(1, min(1000, int(math.ceil(length))) + 1):
                ox, oy = dx * distance, dy * distance
                weight = math.exp(-falloff * distance / max(1.0, length)) / count
                for sx, sy in ((ox, oy), (-ox, -oy)):
                    x0, x1 = max(0, -sx), min(w, w-sx)
                    y0, y1 = max(0, -sy), min(h, h-sy)
                    if x1 > x0 and y1 > y0:
                        glint[y0+sy:y1+sy, x0+sx:x1+sx] += bright[y0:y1, x0:x1] * weight
        out[..., :3] += glint
        return out

    @staticmethod
    def _sparkles(image, p, frame=None, origin=(0, 0)):
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w]
        # Hash per pixel, seed and frame; deterministic across tile boundaries.
        # Absolute-origin offset keeps tiles stable (coordinate hash rather than tile-local RNG).
        q = (xx + int(origin[0])) * 73856093 ^ (yy + int(origin[1])) * 19349663 ^ (int(p.get("seed", 1))+int(frame or 0))*83492791
        q = q.astype(np.uint32)
        q ^= q >> 13; q *= np.uint32(1274126177); q ^= q >> 16
        hits = ((q.astype(np.float64) / 4294967296.0) < float(p.get("density", .02))) & (np.max(image[..., :3], axis=2) > float(p.get("tolerance", 1)))
        # A cross-shaped glint, with a Gaussian centre.
        out = image.copy()
        size = max(1, min(64, int(round(float(p.get("size", 3))))))
        for dy in range(-size, size+1):
            for dx in range(-size, size+1):
                if dx and dy: continue
                weight = math.exp(-max(abs(dx), abs(dy))/max(1, size*.45))
                sy0, sy1 = max(0,-dy), min(h,h-dy); sx0,sx1=max(0,-dx),min(w,w-dx)
                if sy1>sy0 and sx1>sx0:
                    out[sy0+dy:sy1+dy,sx0+dx:sx1+dx,:3] += hits[sy0:sy1,sx0:sx1,None] * weight
        return out

    @staticmethod
    def _godrays(image, p, origin=(0, 0), matte=None):
        decay = max(0.0, float(p.get("decay", .9)))
        if decay == 0.0: return image.copy()
        steps = max(1, min(256, int(p.get("steps", 32))))
        h, w = image.shape[:2]
        yy, xx = np.mgrid[:h, :w].astype(np.float32)
        cx, cy = float(p.get("center_x", w/2))-origin[0], float(p.get("center_y", h/2))-origin[1]
        bright = np.maximum(image[..., :3], 0.0)
        if matte is not None: bright = bright * np.clip(matte[..., 3:4], 0, 1)
        out = image.copy(); acc = np.zeros_like(bright)
        translate = float(p.get("translate", 1.0))
        for i in range(1, steps+1):
            t = translate * i / steps
            sx = np.clip(np.rint(xx + (xx-cx)*t).astype(int), 0, w-1)
            sy = np.clip(np.rint(yy + (yy-cy)*t).astype(int), 0, h-1)
            acc += bright[sy, sx] * (decay ** i)
        out[..., :3] += acc / steps
        return out

    @staticmethod
    def _scanned_grain(image, plate, p, frame=None):
        if plate is None: raise ValueError("ScannedGrain: connect a grain plate")
        h, w = image.shape[:2]; ph, pw = plate.shape[:2]
        # Match each plate channel's scanned mean and variance, wrapping a translated plate per frame.
        seed, frame = int(p.get("seed", 1)), int(frame or 0)
        irregularity = float(p.get("irregularity", 0.0))
        jitter = int(round(irregularity * (seed * 1103515245 + frame * 12345 & 0x7fffffff) / 0x7fffffff * max(ph, pw)))
        sy, sx = (frame + jitter) % ph, (seed*17 + frame + jitter) % pw
        tile = np.tile(plate, (math.ceil((h+sy)/ph), math.ceil((w+sx)/pw), 1))[sy:sy+h, sx:sx+w]
        out = image.copy(); alpha = image[..., 3:4]
        luma = np.maximum(0.0, image[..., :3].mean(axis=2))
        response = np.maximum(np.power(luma, max(.01, float(p.get("response", 1.0)))),
                              float(p.get("minimum", 0.0)))
        preset_gain = {"neutral": 1.0, "35mm": 0.8, "16mm": 1.15, "8mm": 1.45, "reversal": 0.65}.get(p.get("preset", "neutral"), 1.0)
        alpha_gate = alpha if p.get("apply_through_alpha", 0) else 1.0
        for c in range(3):
            noise = tile[..., c] - float(plate[..., c].mean())
            out[..., c] += noise * float(p.get("amount", 1.0)) * response * preset_gain * alpha_gate[..., 0] if isinstance(alpha_gate, np.ndarray) else noise * float(p.get("amount", 1.0)) * response * preset_gain * alpha_gate
        out[..., 3:4] = alpha
        return out

    @staticmethod
    def _distance_transform(features):
        """Exact Euclidean distance to the nearest true pixel, using separable squared EDT."""
        features = np.asarray(features, dtype=bool)
        height, width = features.shape
        inf = np.inf

        def edt_line(cost):
            n = len(cost)
            sites = np.flatnonzero(np.isfinite(cost))
            if not len(sites):
                return np.full(n, inf, np.float64), np.zeros(n, np.int32)
            v = np.empty(len(sites), np.int32)
            z = np.empty(len(sites) + 1, np.float64)
            k = 0; v[0] = sites[0]; z[0] = -inf; z[1] = inf
            for q in sites[1:]:
                q = int(q)
                sep = ((cost[q] + q*q) - (cost[v[k]] + v[k]*v[k])) / (2.0 * (q-v[k]))
                while sep <= z[k]:
                    k -= 1
                    sep = ((cost[q] + q*q) - (cost[v[k]] + v[k]*v[k])) / (2.0 * (q-v[k]))
                k += 1; v[k] = q; z[k] = sep; z[k+1] = inf
            result = np.empty(n, np.float64); closest = np.empty(n, np.int32); k = 0
            for q in range(n):
                while z[k+1] < q: k += 1
                delta = q - v[k]
                result[q] = delta*delta + cost[v[k]]
                closest[q] = v[k]
            return result, closest

        first = np.empty((height, width), np.float64)
        first_x = np.empty((height, width), np.int32)
        for y in range(height):
            first[y], first_x[y] = edt_line(np.where(features[y], 0.0, inf))
        second = np.empty_like(first); nearest_y = np.empty((height, width), np.int32)
        for x in range(width):
            second[:, x], nearest_y[:, x] = edt_line(first[:, x])
        nearest_x = first_x[nearest_y, np.broadcast_to(np.arange(width), (height, width))]
        return np.sqrt(second), nearest_y, nearest_x

    @staticmethod
    def _levelset(source, p):
        if not int(p.get("enabled", 1)):
            return source
        rgba = source.fit(source.data)
        def channel_view(path):
            path = str(path)
            component = path.rsplit(".", 1)[-1].upper()
            index = {"R": 0, "X": 0, "RED": 0, "G": 1, "Y": 1, "GREEN": 1,
                     "B": 2, "Z": 2, "BLUE": 2, "A": 3, "W": 3, "ALPHA": 3}.get(component)
            if index is None:
                raise ValueError(f"LevelSet channel path must name a component: {path}")
            if path.startswith("rgba."):
                return rgba, index, None
            if "." not in path:
                raise ValueError(f"LevelSet channel path must be layer.component: {path}")
            layer_name = path.rsplit(".", 1)[0]
            layer = (source.layers or {}).get(layer_name)
            if layer is None:
                raise ValueError(f"LevelSet: no layer {layer_name!r} on the input")
            return layer.fit(source.data), index, layer_name
        sampled, channel, _ = channel_view(p.get("channel", "rgba.alpha"))
        inside = sampled[..., channel] >= float(p.get("threshold", .5))
        to_inside, nearest_inside_y, nearest_inside_x = Evaluator._distance_transform(inside)
        to_outside, _, _ = Evaluator._distance_transform(~inside)
        maximum_distance = math.hypot(*inside.shape)
        to_inside = np.minimum(to_inside, maximum_distance)
        to_outside = np.minimum(to_outside, maximum_distance)
        # Negative inside, positive outside: a positive matt_limit grows the original matte.
        signed = np.where(inside, -np.maximum(0.0, to_outside - .5),
                          np.maximum(0.0, to_inside - .5)).astype(np.float32)
        pixels = rgba.copy()
        if int(p.get("create_matte", 0)):
            result = (signed < float(p.get("matt_limit", 0.0))).astype(np.float32)
            extrapolate = p.get("extrapolated", "none")
            if extrapolate != "none" and float(p.get("matt_limit", 0.0)) > 0.0:
                expanded = (result > .5) & ~inside
                channels = (0, 1, 2, 3) if extrapolate == "rgba" else (0, 1, 2)
                for c in channels:
                    edge = rgba[..., c][nearest_inside_y, nearest_inside_x]
                    if int(p.get("gradient_extrapolate", 0)) and min(rgba.shape[:2]) > 1:
                        gy, gx = np.gradient(rgba[..., c].astype(np.float64))
                        dy, dx = np.indices(inside.shape)
                        edge = edge + gx[nearest_inside_y, nearest_inside_x] * (dx-nearest_inside_x) + gy[nearest_inside_y, nearest_inside_x] * (dy-nearest_inside_y)
                    pixels[..., c][expanded] = edge[expanded]
        else:
            result = signed
        output_path = str(p.get("output", "rgba.alpha"))
        layers = dict(source.layers or {})
        if output_path != "none":
            output_pixels, output_index, layer_name = channel_view(output_path)
            output_pixels = output_pixels.copy()
            output_pixels[..., output_index] = result
            if layer_name is None:
                pixels[..., output_index] = result
            else:
                old_layer = layers[layer_name]
                layers[layer_name] = Raster(output_pixels, old_layer.data, old_layer.display, old_layer.layers, old_layer.meta)
        gradient = p.get("gradient", "motion")
        if gradient != "none":
            if min(signed.shape) > 1:
                gy, gx = np.gradient(signed.astype(np.float64))
            else:
                gy = np.zeros_like(signed, dtype=np.float64); gx = np.zeros_like(gy)
            direction = np.where(signed < 0, 1.0, -1.0)
            vectors = np.stack((gx * direction, gy * direction), axis=-1)
            length = np.linalg.norm(vectors, axis=-1, keepdims=True)
            vectors = np.divide(vectors, np.maximum(length, 1e-12), out=np.zeros_like(vectors), where=length > 1e-12)
            field = np.zeros_like(pixels)
            field[..., 0:2] = vectors.astype(np.float32)
            field[..., 3] = 1.0
            if gradient == "rgba":
                pixels[..., :2] = field[..., :2]
            else:
                layers["motion"] = Raster(field, source.data, source.display)
        return Raster(pixels, source.data, source.display, layers, source.meta)

    @staticmethod
    def _posterize(image, p):
        # `colors` evenly spaced levels per selected channel, from 0 to 1: round(v * (n - 1)) / (n - 1),
        # clamped to 0..1 (an HDR value lands on the top level).
        levels = float(max(2, int(round(float(p.get("colors", 16))))) - 1)
        out = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgb")]:
            out[..., c] = (np.clip(np.round(image[..., c] * levels), 0.0, levels) / levels).astype(np.float32)
        return out

    @staticmethod
    def _softclip_log_rate(span, room):
        """k > 0 with log(1 + k * span) == k * room, for span > room > 0 (the curve's unit-slope rate)."""
        lo, hi = 1e-9, 1.0
        f = lambda k: math.log1p(k * span) - k * room
        while f(hi) > 0.0 and hi < 1e12:
            hi *= 2.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if f(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    @staticmethod
    def _softclip(image, p):
        # Nuke's four conversions. "logarithmic compress" maps softclip_min..softclip_max onto
        # softclip_min..1 with a logarithmic curve of slope 1 at softclip_min (so values at or below
        # it are untouched and the join is smooth); values above softclip_max keep climbing past 1.
        # A max at or below 1 leaves nothing to compress. The two "preserve hue" modes only act on
        # pixels with a component above softclip_max: brightness is kept by desaturating toward the
        # luma, saturation is kept by scaling the whole pixel down. Alpha is untouched.
        mode = p.get("conversion", "none")
        # The limits are compared as float32, the pixel precision, so a pixel equal to softclip_min stays put.
        lo, hi = float(np.float32(p.get("softclip_min", 0.8))), float(np.float32(p.get("softclip_max", 1.0)))
        rgb = image[..., :3].astype(np.float64)
        if mode == "logarithmic compress":
            span, room = hi - lo, 1.0 - lo
            if span > room > 0.0:
                k = Evaluator._softclip_log_rate(span, room)
                t = np.maximum(rgb - lo, 0.0)
                out_rgb = np.where(rgb > lo, lo + room * np.log1p(k * t) / math.log1p(k * span), rgb)
            else:
                out_rgb = rgb
        elif mode in ("preserve hue and brightness", "preserve hue and saturation"):
            peak = rgb.max(axis=-1, keepdims=True)
            over = peak > hi
            if mode == "preserve hue and saturation":
                out_rgb = np.where(over, rgb * (hi / np.where(over, peak, 1.0)), rgb)
            else:
                luma = 0.2126 * rgb[..., 0:1] + 0.7152 * rgb[..., 1:2] + 0.0722 * rgb[..., 2:3]
                gap = np.where(peak > luma, peak - luma, 1.0)
                keep = np.clip((hi - luma) / gap, 0.0, 1.0)
                out_rgb = np.where(over, luma + (rgb - luma) * keep, rgb)
        else:
            out_rgb = rgb
        return np.concatenate([out_rgb, image[..., 3:4]], axis=2).astype(np.float32)

    @staticmethod
    def _hsv_range_weight(value, lo, hi, rolloff, cyclic=False):
        """1 inside [lo, hi], falling linearly to 0 over `rolloff` outside it. A cyclic range (hue,
        degrees) wraps: lo > hi selects the arc through 360, a span of 360 selects everything. On
        the linear axes (saturation, brightness) a `hi` of 1 or more is open above, so a range that
        ends at the top of 0..1 also takes HDR values."""
        if cyclic:
            if hi - lo >= 360.0:
                return np.ones_like(value)
            lo, hi = lo % 360.0, hi % 360.0
            inside = (value >= lo) & (value <= hi) if lo <= hi else (value >= lo) | (value <= hi)
            def arc(a, b):
                d = np.abs(a - b) % 360.0
                return np.minimum(d, 360.0 - d)
            distance = np.where(inside, 0.0, np.minimum(arc(value, lo), arc(value, hi)))
        else:
            distance = np.maximum(lo - value, 0.0)
            if hi < 1.0:
                distance = np.maximum(distance, value - hi)
        if rolloff <= 0.0:
            return (distance <= 0.0).astype(np.float64)
        return np.clip(1.0 - distance / rolloff, 0.0, 1.0)

    @staticmethod
    def _rgb_to_hsv(rgb):
        """RGB (..., 3) in [0, 1] to (hue degrees, saturation, value), HSVTool's own convention.

        Shared by the per-pixel kernel and by `srccolor`/`dstcolor` (a single triplet), so a
        colour picked in the Viewer and a pixel that matches it land on the same hue exactly.
        """
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        peak, low = rgb.max(axis=-1), rgb.min(axis=-1)
        chroma = peak - low
        safe = np.where(chroma > 0.0, chroma, 1.0)
        hue = np.where(chroma <= 0.0, 0.0, np.where(
            peak == r, 60.0 * (((g - b) / safe) % 6.0),
            np.where(peak == g, 60.0 * ((b - r) / safe + 2.0), 60.0 * ((r - g) / safe + 4.0))))
        sat = np.where(peak > 0.0, chroma / np.where(peak > 0.0, peak, 1.0), 0.0)
        return hue, sat, peak

    @staticmethod
    def _hsv_tool(image, p):
        # HSVTool: RGB -> (hue in degrees, saturation, value), an adjustment weighted by how far
        # inside the hue, saturation and brightness ranges the pixel sits (product of the three
        # weights, each with its own linear rolloff), then back to RGB. Hue rotates by
        # hue_rotation degrees. Saturation and brightness scale by (1 + adjust), or, with the
        # matching "force" toggle, move to the adjust value itself (the definition of Nuke's
        # `saturation`/`brightness` adjustments is inferred, see docs/PARITY_2D.md). Alpha is
        # untouched unless output_alpha is on, when it becomes the combined range weight.
        # `color_replace` (Nuke's Color Replacement, `srccolor`/`dstcolor`) derives the rotation
        # and forced saturation/brightness from the two picked colours instead of the literal
        # rotation/adjust knobs: a pixel that lands at full weight and exactly matches srccolor's
        # hue, saturation and value becomes dstcolor exactly; the existing range/rolloff knobs
        # still gate which pixels that reaches, unchanged.
        rgb = image[..., :3].astype(np.float64)
        hue, sat, val = Evaluator._rgb_to_hsv(rgb)
        weight = (Evaluator._hsv_range_weight(hue, float(p.get("hue_range_min", 0.0)), float(p.get("hue_range_max", 360.0)),
                                              float(p.get("hue_rolloff", 0.0)), cyclic=True)
                  * Evaluator._hsv_range_weight(sat, float(p.get("saturation_range_min", 0.0)),
                                                float(p.get("saturation_range_max", 1.0)),
                                                float(p.get("saturation_rolloff", 0.0)))
                  * Evaluator._hsv_range_weight(val, float(p.get("brightness_range_min", 0.0)),
                                                float(p.get("brightness_range_max", 1.0)),
                                                float(p.get("brightness_rolloff", 0.0))))
        sat_adjust, brt_adjust = float(p.get("sat_adjust", 0.0)), float(p.get("brt_adjust", 0.0))
        rotation = float(p.get("hue_rotation", 0.0))
        set_saturation, set_brightness = bool(p.get("set_saturation")), bool(p.get("set_brightness"))
        if p.get("color_replace"):
            src_rgb = np.array([p.get("srccolor_r", 0.0), p.get("srccolor_g", 0.0), p.get("srccolor_b", 0.0)], np.float64)
            dst_rgb = np.array([p.get("dstcolor_r", 0.0), p.get("dstcolor_g", 0.0), p.get("dstcolor_b", 0.0)], np.float64)
            src_hue, _src_sat, _src_val = Evaluator._rgb_to_hsv(src_rgb)
            dst_hue, dst_sat, dst_val = Evaluator._rgb_to_hsv(dst_rgb)
            rotation = ((float(dst_hue) - float(src_hue) + 180.0) % 360.0) - 180.0
            sat_adjust, brt_adjust = float(dst_sat), float(dst_val)
            set_saturation = set_brightness = True
        new_hue = (hue + rotation * weight) % 360.0
        sat_target = sat_adjust if set_saturation else sat * (1.0 + sat_adjust)
        val_target = brt_adjust if set_brightness else val * (1.0 + brt_adjust)
        new_sat = np.clip(sat + (sat_target - sat) * weight, 0.0, 1.0)
        new_val = np.maximum(val + (val_target - val) * weight, 0.0)
        out_rgb = np.stack([new_val - new_val * new_sat * np.clip(np.minimum((n + new_hue / 60.0) % 6.0,
                                                                            4.0 - (n + new_hue / 60.0) % 6.0), 0.0, 1.0)
                            for n in (5.0, 3.0, 1.0)], axis=-1)
        untouched = (rotation == 0.0 and sat_adjust == 0.0 and brt_adjust == 0.0
                     and not set_saturation and not set_brightness)
        if untouched:
            out_rgb = rgb
        alpha = weight[..., None] if p.get("output_alpha") else image[..., 3:4]
        return np.concatenate([out_rgb, alpha], axis=2).astype(np.float32)

    @staticmethod
    def _blend(layers, p):
        """Weighted average of the wired inputs, `layers` = [(slot index, array)]. With normalize on the
        weights are divided by their sum (a zero sum leaves black); off, the weighted sum is returned.
        Only the selected `channels` are blended; the rest come from the first input."""
        weights = [np.float32(p.get(f"weight{i}", 1.0)) for i, _ in layers]
        total = np.float32(sum(weights))
        blended = sum(w * image for w, (_, image) in zip(weights, layers))
        if p.get("normalize", 1):
            blended = blended / total if total != 0 else np.zeros_like(blended)
        if p.get("fringe"):
            # Unpremultiplied blend: each input's colour is divided by its own alpha, averaged with the
            # weights as straight colour, then multiplied by the blended alpha. A low-coverage edge
            # pixel therefore keeps its full colour in the average instead of being scaled down by its
            # own alpha first, so the edge leans toward the thinner input's colour.
            straight = sum(w * np.divide(image[..., :3], image[..., 3:4], out=image[..., :3].copy(),
                                         where=np.abs(image[..., 3:4]) > 1e-6)
                           for w, (_, image) in zip(weights, layers))
            if p.get("normalize", 1):
                straight = straight / total if total != 0 else np.zeros_like(straight)
            blended = blended.copy()
            blended[..., :3] = straight * blended[..., 3:4]
        out = layers[0][1].copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c] = blended[..., c]
        return out.astype(np.float32)

    @staticmethod
    def _copy_rectangle(a, b, p, origin=(0, 0)):
        # A's selected channels replace B's inside the area box: left/top edges area_x/area_y, right
        # and bottom edges area_r/area_t, in canvas pixels (`origin` is where this array sits). A
        # pixel is inside when its centre is, so integer edges copy exact whole pixels. `softness`
        # fades the copy to B over that fraction of half the shorter side, measured inward from
        # every edge; 0 is a hard edge.
        height, width = b.shape[:2]
        x0, y0 = float(p.get("area_x", 0.0)), float(p.get("area_y", 0.0))
        x1, y1 = float(p.get("area_r", 0.0)), float(p.get("area_t", 0.0))
        px = np.arange(width, dtype=np.float64) + int(origin[0]) + 0.5
        py = np.arange(height, dtype=np.float64) + int(origin[1]) + 0.5
        dx = np.minimum(px - x0, x1 - px)
        dy = np.minimum(py - y0, y1 - py)
        depth = np.minimum(dx[None, :], dy[:, None])
        band = float(p.get("softness", 0.0)) * 0.5 * max(0.0, min(x1 - x0, y1 - y0))
        gate = np.clip(depth / band, 0.0, 1.0) if band > 0.0 else (depth > 0.0).astype(np.float64)
        gate = gate[..., None].astype(np.float32)
        out = b.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            out[..., c:c + 1] = b[..., c:c + 1] + (a[..., c:c + 1] - b[..., c:c + 1]) * gate
        return out

    @staticmethod
    def _bilinear_gather(image, qx, qy):
        """Bilinear samples of `image` at continuous positions (`qx`, `qy`), where pixel (i, j)
        has its centre at (i + 0.5, j + 0.5). Positions past the frame clamp to the edge pixel,
        the "edge" padding the other padded filters use."""
        height, width = image.shape[:2]
        u, v = qx - 0.5, qy - 0.5
        x0, y0 = np.floor(u), np.floor(v)
        fx, fy = (u - x0)[..., None], (v - y0)[..., None]
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        x1, y1 = np.clip(x0 + 1, 0, width - 1), np.clip(y0 + 1, 0, height - 1)
        x0, y0 = np.clip(x0, 0, width - 1), np.clip(y0, 0, height - 1)
        return ((image[y0, x0] * (1 - fx) + image[y0, x1] * fx) * (1 - fy)
                + (image[y1, x0] * (1 - fx) + image[y1, x1] * fx) * fy)

    @staticmethod
    def _dirblur(image, p):
        # Nuke's DirBlur. All three types average bilinear samples taken along a path through each
        # pixel; the weights sum to 1 so flat regions and total energy are preserved.
        #   linear: a box of `length` pixels along `angle` (degrees, counter-clockwise from +x with
        #           y up, as in Nuke). Taps sit on whole-pixel steps along the line, the two end
        #           taps weighted by how much of their pixel the box covers, so length 8 is exactly
        #           eight pixels of coverage and length 1 or less is the identity. Symmetric about
        #           the pixel, and the reach is ceil(length / 2) + 1 pixels (the tile region rule).
        #   zoom:   samples on the ray through the centre, scale 1 +/- length / 200.
        #   radial: samples on the circle about the centre, +/- angle / 2 degrees.
        # The centre is in this array's own pixel coordinates (see `_filtered_pixels`).
        kind = p.get("blur_type", "linear")
        height, width = image.shape[:2]
        if kind == "linear":
            length = abs(float(p.get("length", 0.0)))
            if length <= 1.0:
                return image.copy()
            half = length / 2.0
            reach = int(math.floor(half + 0.5))
            angle = math.radians(float(p.get("angle", 0.0)))
            dx, dy = math.cos(angle), -math.sin(angle)
            pad = reach + 1
            padded = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode="edge").astype(np.float64)
            acc = np.zeros(image.shape, dtype=np.float64)
            total = 0.0
            for t in range(-reach, reach + 1):
                weight = min(t + 0.5, half) - max(t - 0.5, -half)
                if weight <= 0.0:
                    continue
                ox, oy = round(t * dx, 9), round(t * dy, 9)
                ix, iy = int(math.floor(ox)), int(math.floor(oy))
                fx, fy = ox - ix, oy - iy
                for (sx, sy, w) in ((ix, iy, (1 - fx) * (1 - fy)), (ix + 1, iy, fx * (1 - fy)),
                                    (ix, iy + 1, (1 - fx) * fy), (ix + 1, iy + 1, fx * fy)):
                    if w:
                        acc += (weight * w) * padded[pad + sy:pad + sy + height, pad + sx:pad + sx + width]
                total += weight
            acc /= total
        elif kind in ("zoom", "radial"):
            cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
            gy, gx = np.mgrid[0:height, 0:width]
            rx, ry = gx + 0.5 - cx, gy + 0.5 - cy
            far = math.hypot(max(abs(0.0 - cx), abs(width - cx)), max(abs(0.0 - cy), abs(height - cy)))
            if kind == "zoom":
                span = abs(float(p.get("length", 0.0))) / 100.0
                if span <= 0.0:
                    return image.copy()
                count = int(min(96, max(2, math.ceil(far * span) + 1)))
                steps = [1.0 + s for s in np.linspace(-span / 2.0, span / 2.0, count)]
                positions = [(cx + rx * s, cy + ry * s) for s in steps]
            else:
                sweep = math.radians(abs(float(p.get("angle", 0.0))))
                if sweep <= 0.0:
                    return image.copy()
                count = int(min(96, max(2, math.ceil(far * sweep) + 1)))
                positions = []
                for a in np.linspace(-sweep / 2.0, sweep / 2.0, count):
                    c, s = math.cos(a), math.sin(a)
                    positions.append((cx + rx * c - ry * s, cy + rx * s + ry * c))
            acc = np.zeros(image.shape, dtype=np.float64)
            for qx, qy in positions:
                acc += Evaluator._bilinear_gather(image.astype(np.float64), qx, qy)
            acc /= len(positions)
        else:
            raise ValueError(f"Unknown DirBlur blur_type {kind!r}")
        result = image.copy()
        for c in Evaluator._CHANNEL_SETS[p.get("channels", "rgba")]:
            result[..., c] = acc[..., c].astype(np.float32)
        return result

    @staticmethod
    def _mirror(image, p):
        # Flip about the format centre: a pixel at x moves to width-1-x (flip_x) and/or
        # y moves to height-1-y (flip_y). No mask/mix-independent box growth -- it is a pure
        # in-place reindex, like Invert, just spatial instead of per-channel.
        out = image
        if p.get("flip_x", 0):
            out = out[:, ::-1, :]
        if p.get("flip_y", 0):
            out = out[::-1, :, :]
        return np.ascontiguousarray(out, dtype=np.float32)

    @staticmethod
    def _dissolve(a, b, p):
        # which=0 -> A, which=1 -> B, values between cross-fade linearly.
        which = np.float32(p.get("which", 0.0))
        return a * (1 - which) + b * which

    @staticmethod
    def _copy_channels(a, b, p):
        out = b.copy()
        index = {"r": 0, "g": 1, "b": 2, "a": 3}
        for out_channel, param in (("r", "copy_red"), ("g", "copy_green"),
                                   ("b", "copy_blue"), ("a", "copy_alpha")):
            source = p.get(param, "none")
            if source == "none":
                continue
            _, src_channel = source.split(".")
            out[..., index[out_channel]] = a[..., index[src_channel]]
        return out

    @staticmethod
    def _difference_key(a, b, p):
        """Nuke's Difference keyer: alpha from the largest per-channel colour difference between A
        and B (max, not mean -- a change in any one channel should be enough to key, the same
        reasoning `_merge_op`'s own `difference` uses per-channel rather than collapsing to one
        number first), `offset` and `gain` shaping it, output is B's colour with the new alpha."""
        diff = np.max(np.abs(a[..., :3] - b[..., :3]), axis=-1, keepdims=True)
        alpha = np.clip((diff - p.get("offset", 0.0)) * p.get("gain", 1.0), 0.0, 1.0)
        return np.concatenate([b[..., :3], alpha], axis=2).astype(np.float32)

    @staticmethod
    def _channel_merge(a, b, p):
        index = {"r": 0, "g": 1, "b": 2, "a": 3}
        a_idx = index[p.get("a_channel", "A.a").split(".")[1]]
        b_idx = index[p.get("b_channel", "B.a").split(".")[1]]
        out_idx = {"R": 0, "G": 1, "B": 2, "A": 3}[p.get("out_channel", "A")]
        combined = Evaluator._channel_merge_op(p.get("operation", "over"),
                                               a[..., a_idx:a_idx + 1], b[..., b_idx:b_idx + 1])
        out = b.copy()
        out[..., out_idx:out_idx + 1] = combined
        return out

    @staticmethod
    def _mask_matte(mask, channel="alpha"):
        """The matte a mask input supplies, as an (h, w, 1) array: its alpha by default, or the red,
        green or blue channel, or its Rec. 709 luminance (the `mask_channel` choice)."""
        if channel == "luminance":
            return (0.2126 * mask[..., 0:1] + 0.7152 * mask[..., 1:2] + 0.0722 * mask[..., 2:3]).astype(np.float32)
        return mask[..., {"red": 0, "green": 1, "blue": 2}.get(channel, 3):][..., :1]

    @staticmethod
    def _light_mixer(source, mask, p):
        """LightMixer: the beauty with each light group's `light.<group>` layer (Render3D's `lights` pass, or an EXR's
        own) scaled by a gain and a colour: `beauty + sum((gain * colour - 1) * layer)`. Everything the layers do not
        hold (emission, an unrelit splat, layers beyond the eight slots) stays as it was; all gains at 1 and colours at
        white add exact zeros, so the beauty comes back bit for bit, and a gain of 0 removes that layer's light.
        Slot `n` names its group in `lm_group{n}`; a blank slot takes the next `light.*` layer (in name order) that no
        slot names. `mask` and `mix` gate the result like Grade's. Whole-image path only: it reads named layers."""
        layers = source.layers or {}
        found = sorted(name for name in layers if name.startswith("light."))
        if not found:
            raise ValueError("LightMixer: connect a Render3D whose passes include 'lights', or an EXR with light.* layers")
        named = {}
        for n in range(1, LIGHT_MIXER_SLOTS + 1):
            label = str(p.get(f"lm_group{n}", "")).strip()
            if label:
                layer = label if label.startswith("light.") else f"light.{label}"
                if layer not in layers:
                    raise ValueError(f"LightMixer: there is no {layer} layer (the input has {', '.join(found)})")
                named[n] = layer
        spare = [name for name in found if name not in named.values()]
        pixels = source.pixels
        total = np.zeros(pixels.shape[:2] + (3,), np.float32)
        for n in range(1, LIGHT_MIXER_SLOTS + 1):
            layer = named.get(n) or (spare.pop(0) if spare else None)
            if layer is None:
                continue
            gain = float(p.get(f"lm_gain{n}", 1.0))
            tint = np.array([float(p.get(f"lm_{c}{n}", 1.0)) for c in ("red", "green", "blue")], np.float32)
            factor = np.float32(gain) * tint - np.float32(1.0)
            if np.any(factor):
                total += layers[layer].fit(source.data)[..., :3] * factor
        mixed = pixels.copy()
        mixed[..., :3] += total
        if mask is not None or float(p.get("mix", 1.0)) != 1.0:
            mixed = Evaluator._apply_mask_mix(pixels, mixed, None if mask is None else mask.fit(source.data),
                                              float(p.get("mix", 1.0)))
        return Raster(mixed, source.data, source.display, source.layers, source.meta)

    @staticmethod
    def _apply_mask_mix(source, filtered, mask, mix, mask_channel="alpha"):
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
            gate = (Evaluator._mask_matte(mask, mask_channel) * np.float32(mix)).astype(np.float32)
        return (filtered * gate + source * (1.0 - gate)).astype(np.float32)

    # --- Draw-menu generators (Ramp, Radial, Rectangle, Noise, Text) -----------------------
    #
    # `_draw_shape` is the one function both `_windowed_kernel` (full frame, at (0, 0, width,
    # height)) and `tileexec._evaluate_tile_kernel` (one tile, at (region.x, region.y, region.width,
    # region.height)) call, with no other path to a pixel: a tile seam can only be identical to the
    # full-frame reference if it runs the *same* code, not a second implementation kept in sync by
    # hand. Every shape returns straight premultiplied RGBA covering exactly (x0, y0, w, h) in
    # absolute canvas coordinates -- column x0 is pixel index x0, not a half-pixel-centred sample --
    # matching Checker's own `xx // size` integer-index convention.

    @staticmethod
    def _draw_shape(kind, p, x0, y0, w, h, frame):
        if kind == "Ramp":
            return Evaluator._ramp_shape(p, x0, y0, w, h)
        if kind == "Radial":
            return Evaluator._radial_shape(p, x0, y0, w, h)
        if kind == "Rectangle":
            return Evaluator._rectangle_shape(p, x0, y0, w, h)
        if kind == "Noise":
            return Evaluator._noise_shape(p, x0, y0, w, h)
        if kind == "Text":
            return Evaluator._text_shape(p, x0, y0, w, h)
        if kind == "Grid":
            return Evaluator._grid_shape(p, x0, y0, w, h)
        raise ValueError(f"No draw shape for {kind}")

    @staticmethod
    def _composite_shape_over(shape, background):
        """Premultiplied "over": the shape is drawn over its optional background, Nuke's own
        Draw-node convention. `shape`'s alpha is where the shape itself is opaque; the background
        shows through everywhere the shape is not."""
        shape_alpha = shape[..., 3:4]
        return (shape + background * (1.0 - shape_alpha)).astype(np.float32)

    @staticmethod
    def _ramp_shape(p, x0, y0, w, h):
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        p0x, p0y = float(p["p0_x"]), float(p["p0_y"])
        p1x, p1y = float(p["p1_x"]), float(p["p1_y"])
        dx, dy = p1x - p0x, p1y - p0y
        denom = dx * dx + dy * dy
        # p0 == p1 is a degenerate ramp with no direction; Nuke holds colour0 everywhere rather
        # than dividing by zero.
        t = np.zeros_like(xx) if denom < 1e-12 else ((xx - p0x) * dx + (yy - p0y) * dy) / denom
        t = np.clip(t, 0.0, 1.0)[..., None]
        c0 = np.array([p["color0_red"], p["color0_green"], p["color0_blue"], p["color0_alpha"]])
        c1 = np.array([p["color1_red"], p["color1_green"], p["color1_blue"], p["color1_alpha"]])
        straight = c0[None, None, :] * (1.0 - t) + c1[None, None, :] * t
        alpha = straight[..., 3:4]
        return np.concatenate([straight[..., :3] * alpha, alpha], axis=-1).astype(np.float32)

    @staticmethod
    def _box_edge_distance(p, x0, y0, w, h):
        """Signed distance (pixels) from the nearest edge of `box_*`: positive inside, negative
        outside. The box's valid columns/rows are `[box_x, box_x + box_width)`, the same half-open
        convention `Region` uses everywhere else in this codebase."""
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        bx, by = float(p["box_x"]), float(p["box_y"])
        bw, bh = float(p["box_width"]), float(p["box_height"])
        return np.minimum(np.minimum(xx - bx, bx + bw - 1.0 - xx),
                          np.minimum(yy - by, by + bh - 1.0 - yy)), bw, bh

    @staticmethod
    def _paint_coverage(p, coverage):
        """A flat colour (`red`/`green`/`blue`/`alpha`) modulated by a 0..1 coverage mask,
        premultiplied. Shared by Radial and Rectangle, which differ only in how coverage is shaped."""
        color = np.array([p["red"], p["green"], p["blue"], p["alpha"]])
        alpha = (coverage * color[3]).astype(np.float32)
        return np.concatenate([alpha[..., None] * color[:3], alpha[..., None]], axis=-1).astype(np.float32)

    @staticmethod
    def _rectangle_shape(p, x0, y0, w, h):
        dist, bw, bh = Evaluator._box_edge_distance(p, x0, y0, w, h)
        softness = float(p.get("softness", 0.0)) * min(bw, bh) / 2.0
        coverage = (dist >= 0).astype(np.float64) if softness < 1e-9 else np.clip(dist / softness, 0.0, 1.0)
        return Evaluator._paint_coverage(p, coverage)

    @staticmethod
    def _grid_lines(index, extent, spacing, number, offset, line_width):
        """Line coverage (0..1) along one axis for absolute pixel indices `index`. A line starts at
        `offset + k * step` for every integer k and covers `line_width` pixels from there, so with
        whole-pixel step, offset and width the covered indices are exactly those where
        `(index - offset) mod step < line_width`. `number` above zero replaces the spacing with
        `extent / number`, which puts `number` lines across the format."""
        step = float(extent) / float(number) if number > 0 else float(spacing)
        phase = np.mod(index - float(offset), step)
        return np.clip(float(line_width) - phase, 0.0, 1.0)

    @staticmethod
    def _grid_shape(p, x0, y0, w, h):
        cols = Evaluator._grid_lines(np.arange(x0, x0 + w, dtype=np.float64), p["width"], p["spacing_x"],
                                     p["number_x"], p["grid_offset_x"], p["line_width"])
        rows = Evaluator._grid_lines(np.arange(y0, y0 + h, dtype=np.float64), p["height"], p["spacing_y"],
                                     p["number_y"], p["grid_offset_y"], p["line_width"])
        coverage = np.maximum(rows[:, None], cols[None, :])
        return Evaluator._paint_coverage(p, coverage)

    @staticmethod
    def _radial_shape(p, x0, y0, w, h):
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        bx, by = float(p["box_x"]), float(p["box_y"])
        bw, bh = float(p["box_width"]), float(p["box_height"])
        cx, cy = bx + (bw - 1.0) / 2.0, by + (bh - 1.0) / 2.0
        rx, ry = max(bw / 2.0, 1e-9), max(bh / 2.0, 1e-9)
        d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
        softness = float(p.get("softness", 0.0))
        coverage = (d < 1.0).astype(np.float64) if softness < 1e-9 else np.clip((1.0 - d) / softness, 0.0, 1.0)
        return Evaluator._paint_coverage(p, coverage)

    @staticmethod
    def _hash_lattice(ix, iy, iz, seed):
        """A deterministic pseudo-random value in [0, 1) per integer lattice point. Pure integer
        arithmetic (no `random`/`np.random`, whose sequences are seed-state, not coordinate,
        driven): the same (ix, iy, iz, seed) always hashes to the same value, on any machine,
        whether it is asked for as a whole canvas or one tile at a time."""
        h = (ix.astype(np.int64) * np.int64(374761393) + iy.astype(np.int64) * np.int64(668265263)
            + np.int64(int(iz)) * np.int64(2147483647) + np.int64(int(seed)) * np.int64(2654435761))
        h = h & np.int64(0xffffffff)
        h = ((h ^ (h >> 13)) * np.int64(1274126177)) & np.int64(0xffffffff)
        h = h ^ (h >> 16)
        return h.astype(np.float64) / 4294967295.0

    @staticmethod
    def _value_noise(gx, gy, gz, seed):
        ix0, iy0 = np.floor(gx).astype(np.int64), np.floor(gy).astype(np.int64)
        fx, fy = gx - ix0, gy - iy0
        iz = math.floor(gz)
        # Quintic smoothstep (Perlin's improved curve): zero first and second derivative at 0/1,
        # so octave boundaries never show a slope discontinuity.
        sx = fx * fx * fx * (fx * (fx * 6 - 15) + 10)
        sy = fy * fy * fy * (fy * (fy * 6 - 15) + 10)
        v00 = Evaluator._hash_lattice(ix0, iy0, iz, seed)
        v10 = Evaluator._hash_lattice(ix0 + 1, iy0, iz, seed)
        v01 = Evaluator._hash_lattice(ix0, iy0 + 1, iz, seed)
        v11 = Evaluator._hash_lattice(ix0 + 1, iy0 + 1, iz, seed)
        top = v00 + sx * (v10 - v00)
        bottom = v01 + sx * (v11 - v01)
        return top + sy * (bottom - top)

    @staticmethod
    def _noise_shape(p, x0, y0, w, h):
        xs = np.arange(x0, x0 + w, dtype=np.float64)
        ys = np.arange(y0, y0 + h, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        size = max(float(p.get("size", 64.0)), 1e-6)
        z = float(p.get("z_slice", 0.0))
        seed = int(p.get("seed", 0))
        octaves = max(1, int(p.get("octaves", 1)))
        lacunarity = float(p.get("lacunarity", 2.0))
        gain = float(p.get("gain", 0.5))
        gamma = max(float(p.get("gamma", 1.0)), 1e-6)
        total = np.zeros_like(xx)
        amplitude, freq, amp_sum = 1.0, 1.0 / size, 0.0
        for octave in range(octaves):
            total = total + amplitude * Evaluator._value_noise(xx * freq, yy * freq, z, seed + octave * 101)
            amp_sum += amplitude
            amplitude *= gain
            freq *= lacunarity
        value = np.clip(total / max(amp_sum, 1e-9), 0.0, 1.0) ** (1.0 / gamma)
        value = value.astype(np.float32)
        alpha = np.ones_like(value)
        return np.concatenate([np.repeat(value[..., None], 3, axis=-1), alpha[..., None]], axis=-1).astype(np.float32)

    @staticmethod
    def _text_shape(p, x0, y0, w, h):
        """Qt's own text rasteriser (QPainter/QFont on an offscreen QImage), no new dependency.

        Font-availability limit: `font` is a family name resolved through Qt's font database at
        render time, exactly like Nuke's own font picker resolves a family on the machine running
        Nuke. Which families are installed is a machine property NodeBased does not control or
        embed -- a comp that names a font missing on another artist's machine renders in whatever
        Qt substitutes there, the same portability limit Nuke's text tools have always had.
        """
        from PySide6.QtCore import QRectF, Qt
        from PySide6.QtGui import QColor, QFont, QImage, QPainter
        from PySide6.QtWidgets import QApplication

        if QApplication.instance() is None:
            QApplication(["nodebased"])
        w, h = max(1, int(w)), max(1, int(h))
        image = QImage(w, h, QImage.Format.Format_RGBA8888)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        # Paint in absolute canvas coordinates; the translate is what makes a tile's render of a
        # box that straddles its edge identical to the full-frame render cropped to that tile.
        painter.translate(-x0, -y0)
        family = p.get("font") or ""
        font = QFont(family) if family else QFont()
        font.setPointSizeF(max(1.0, float(p.get("font_size", 48.0))))
        painter.setFont(font)
        color = QColor.fromRgbF(float(np.clip(p.get("red", 1.0), 0.0, 1.0)),
                                float(np.clip(p.get("green", 1.0), 0.0, 1.0)),
                                float(np.clip(p.get("blue", 1.0), 0.0, 1.0)),
                                float(np.clip(p.get("alpha", 1.0), 0.0, 1.0)))
        painter.setPen(color)
        box = QRectF(float(p["box_x"]), float(p["box_y"]), float(p["box_width"]), float(p["box_height"]))
        justify_flags = {"left": Qt.AlignmentFlag.AlignLeft, "center": Qt.AlignmentFlag.AlignHCenter,
                         "right": Qt.AlignmentFlag.AlignRight}
        align = justify_flags.get(p.get("justify", "left"), Qt.AlignmentFlag.AlignLeft)
        painter.drawText(box, int(align | Qt.AlignmentFlag.AlignVCenter) | int(Qt.TextFlag.TextWordWrap),
                         str(p.get("message", "")))
        painter.end()
        stride = image.bytesPerLine()
        raw = np.frombuffer(bytes(image.constBits()), dtype=np.uint8).reshape(h, stride)
        straight = raw[:, :w * 4].reshape(h, w, 4).astype(np.float32) / 255.0
        alpha = straight[..., 3:4]
        return np.concatenate([straight[..., :3] * alpha, alpha], axis=-1).astype(np.float32)

    @staticmethod
    def _grade(image, p):
        frame = image.copy()
        frame[..., :3] = frame[..., :3] * (2.0 ** p["exposure"]) * p["multiply"] + p["offset"] * frame[..., 3:4]
        return frame

    @staticmethod
    def _histogram_levels(image, p):
        span = float(p["white"]) - float(p["black"])
        if abs(span) < 1e-12:
            raise ValueError("Histogram white input must differ from black input")
        gamma = max(1e-6, float(p["gamma"]))
        out = image.copy()
        x = np.clip((image[..., :3] - float(p["black"])) / span, 0.0, 1.0)
        x = np.power(x, 1.0 / gamma)
        out[..., :3] = float(p["black_out"]) + x * (float(p["white_out"]) - float(p["black_out"]))
        return out

    @staticmethod
    def _hist_eq(image, p):
        out = image.copy()
        if p.get("hist_eq_mode", "luminance") == "luminance":
            luma = np.sum(image[..., :3] * np.array([0.2126, 0.7152, 0.0722], np.float32), axis=-1)
            lo, hi = float(np.min(luma)), float(np.max(luma))
            if hi > lo:
                hist, edges = np.histogram(luma, bins=256, range=(lo, hi))
                cdf = np.cumsum(hist, dtype=np.float64)
                cdf = (cdf - cdf[0]) / max(float(cdf[-1] - cdf[0]), 1.0)
                mapped = np.interp(luma, edges[:-1], cdf).astype(np.float32)
                scale = np.divide(mapped, luma, out=np.ones_like(luma), where=np.abs(luma) > 1e-8)
                out[..., :3] *= scale[..., None]
        else:
            for c in range(3):
                values = image[..., c]
                lo, hi = float(np.min(values)), float(np.max(values))
                if hi > lo:
                    hist, edges = np.histogram(values, bins=256, range=(lo, hi))
                    cdf = np.cumsum(hist, dtype=np.float64)
                    cdf = (cdf - cdf[0]) / max(float(cdf[-1] - cdf[0]), 1.0)
                    out[..., c] = np.interp(values, edges[:-1], cdf)
        return out

    @staticmethod
    def _min_color(image, mode="minimum"):
        luma = image[..., :3] @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        y, x = np.unravel_index(int(np.argmin(luma) if mode == "minimum" else np.argmax(luma)), luma.shape)
        return image[y, x].copy(), (x, y)

    @staticmethod
    def _sample_line(image, start, end, origin=(0, 0)):
        count = max(image.shape[:2])
        x = np.clip(np.rint(np.linspace(start[0] - origin[0], end[0] - origin[0], count)).astype(int), 0, image.shape[1] - 1)
        y = np.clip(np.rint(np.linspace(start[1] - origin[1], end[1] - origin[1], count)).astype(int), 0, image.shape[0] - 1)
        return image[y, x].copy()

    @staticmethod
    def _match_grade(source, target, p):
        # target is None once the node is baked (match_analyzed) and the reference is
        # disconnected: the measured gain/offset already live in the knobs, Nuke's baked-knob
        # workflow, so the live target statistics are neither read nor required.
        out = source.copy()
        baked = bool(p.get("match_analyzed", 0))
        for c, channel in enumerate("rgb"):
            s = source[..., c]
            sm, ss = float(np.mean(s)), float(np.std(s))
            if baked:
                gain = float(p.get(f"grade_gain_{channel}", 1.0))
                result = s * gain + float(p.get(f"grade_offset_{channel}", 0.0)) + float(p.get(f"grade_lift_{channel}", 0.0))
            else:
                t = target[..., c]
                tm, ts = float(np.mean(t)), float(np.std(t))
                gain = (ts / ss if ss > 1e-12 else 1.0) * float(p.get(f"grade_gain_{channel}", 1.0))
                result = (s - sm) * gain + tm + float(p.get(f"grade_lift_{channel}", 0.0)) + float(p.get(f"grade_offset_{channel}", 0.0))
            gamma = max(1e-6, float(p.get(f"grade_gamma_{channel}", 1.0)))
            if gamma != 1.0:
                result = np.sign(result) * np.power(np.abs(result), 1.0 / gamma)
            out[..., c] = result
        return out

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
    def _log2lin(image, p):
        x = image.copy(); rgb = x[..., :3]
        black, white = float(p["black"]), float(p["white"])
        span = white - black
        if abs(span) < 1e-12: span = 1e-12
        if p["log_direction"] == "log to lin":
            rgb[:] = np.sign(rgb - black) * np.abs((rgb - black) / span) ** max(float(p["gamma"]), 1e-6)
        else:
            rgb[:] = black + np.sign(rgb) * np.abs(rgb) ** (1.0 / max(float(p["gamma"]), 1e-6)) * span
        return x

    @staticmethod
    def _ploglin(image, p):
        x = image.copy(); rgb = x[..., :3]
        lr, cr = float(p["linear_reference"]), float(p["log_reference"])
        d = max(float(p["density_per_code_value"]), 1e-9)
        gamma = max(float(p["negative_gamma"]), 1e-6)
        # Signed density mapping is stable for negative scene values and has exact reference anchors.
        delta = (rgb - cr) * d
        mapped = lr * np.power(10.0, np.clip(delta, -30, 30))
        rgb[:] = np.where(rgb < 0, -lr * np.power(10.0, np.clip((np.abs(rgb) - cr) * d, -30, 30)) ** gamma, mapped)
        return x

    @staticmethod
    def _crosstalk(image, p):
        # Nuke's `unpremult` divides the curves' input by a chosen channel (usually alpha) before
        # the 3x3 lookup and multiplies back afterward, so the curves see straight colour instead
        # of premultiplied; `fringe` then limits the whole effect to partial-alpha edge pixels
        # (0 < alpha < 1), leaving solid interior and fully transparent pixels untouched.
        from . import colorcurves
        src = image[..., :3]
        channel_index = {"red": 0, "green": 1, "blue": 2, "alpha": 3}.get(p.get("xt_unpremult", "none"))
        if channel_index is not None:
            divisor = image[..., channel_index:channel_index + 1]
            safe = np.abs(divisor) > 1e-6
            work = np.divide(src, divisor, out=src.copy(), where=safe)
        else:
            work = src
        out = np.zeros_like(work)
        for oi, oc in enumerate("rgb"):
            for si, sc in enumerate("rgb"):
                v = work[..., si]
                raw = p.get(f"xt_curve_{oc}_{sc}")
                if raw is None:
                    vals = np.array([p[f"xt_{oc}_{sc}_{i}"] for i in range(3)], dtype=np.float32)
                    mapped = np.interp(v, (0.0, 0.5, 1.0), vals)
                    mapped = np.where(v < 0, vals[0] + v * (vals[1] - vals[0]) * 2,
                                      np.where(v > 1, vals[2] + (v - 1) * (vals[2] - vals[1]) * 2, mapped))
                else:
                    curve = colorcurves.decode(raw)
                    mapped = colorcurves.evaluate_array(curve, v)
                out[..., oi] += mapped
        if channel_index is not None:
            out = np.where(safe, out * divisor, src)
        if p.get("xt_fringe", False):
            alpha = image[..., 3:4]
            edge = (alpha > 1e-6) & (alpha < 1.0 - 1e-6)
            out = np.where(edge, out, src)
        result = image.copy(); result[..., :3] = out
        return result

    @staticmethod
    def _toe(image, p):
        result = image.copy(); rgb = result[..., :3]
        knee = float(np.clip(p["toe"], 1e-6, 0.999999)); lift = float(p["toe_lift"])
        low = (rgb >= 0) & (rgb < knee)
        shape = np.square(np.maximum(1.0 - rgb / knee, 0.0))
        rgb[:] = np.where(low, rgb + lift * shape, rgb)
        return result

    @staticmethod
    def _expression(image, second, p, frame, origin=(0, 0), canvas_size=None):
        from .ops2d_expression import evaluate_channels
        try:
            return evaluate_channels(image, second, p, frame, origin, canvas_size)
        except ValueError as error:
            raise ValueError(f"Expression node: {error}") from None

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
    def _reformat_place(target_w, target_h, src_w, src_h, resize_type, center, turn):
        """Scale and offset placing a `src_w`x`src_h` source into a `target_w`x`target_h` format,
        before flip/flop/turn are applied. Shared by `_reformat` (the kernel) and
        `_reformat_transformed_window` (the preserve-bounding-box case) so the two formulas
        cannot drift apart, the same reason `_transform`/`_transformed_window` share their math.

        `turn` solves the resize against the *swapped* working format -- Nuke's own turn knob
        also treats width and height as exchanged before deciding the resize -- so a turned
        Reformat still fits/fills correctly instead of using the unturned format's aspect.
        """
        tw, th = (target_h, target_w) if turn else (target_w, target_h)
        sx0 = tw / src_w if src_w else 1.0
        sy0 = th / src_h if src_h else 1.0
        if resize_type == "none":
            scale_x = scale_y = 1.0
        elif resize_type == "width":
            scale_x = scale_y = sx0
        elif resize_type == "height":
            scale_x = scale_y = sy0
        elif resize_type == "fit":
            scale_x = scale_y = min(sx0, sy0)
        elif resize_type == "fill":
            scale_x = scale_y = max(sx0, sy0)
        elif resize_type == "distort":
            scale_x, scale_y = sx0, sy0
        else:
            raise ValueError(f"Unknown Reformat resize type: {resize_type}")
        if center:
            offset_x = (tw - src_w * scale_x) / 2.0
            offset_y = (th - src_h * scale_y) / 2.0
        else:
            offset_x = offset_y = 0.0
        return scale_x, scale_y, offset_x, offset_y, tw, th

    @staticmethod
    def _reformat_transformed_window(box, src_display_w, src_display_h, target_w, target_h, p):
        """Forward-map the corners of `box` through the same placement `_reformat` inverts, for
        the preserve-bounding-box case. Exact inverse of `_reformat`'s sampling map."""
        if box.is_empty:
            return box
        resize_type, center = p["resize_type"], p["center"]
        flip, flop, turn = p["flip"], p["flop"], p["turn"]
        scale_x, scale_y, offset_x, offset_y, _tw, _th = Evaluator._reformat_place(
            target_w, target_h, src_display_w, src_display_h, resize_type, center, turn)
        xs, ys = [], []
        for bx, by in ((box.x, box.y), (box.right, box.y), (box.x, box.bottom), (box.right, box.bottom)):
            px, py = bx * scale_x + offset_x, by * scale_y + offset_y
            fx, fy = (py, px) if turn else (px, py)
            if flop:
                fx = target_w - fx
            if flip:
                fy = target_h - fy
            xs.append(fx)
            ys.append(fy)
        support = {"nearest": 1, "bilinear": 1, "cubic": 2}.get(p.get("filter", "nearest"), 2)
        left, top = math.floor(min(xs)) - support, math.floor(min(ys)) - support
        right, bottom = math.ceil(max(xs)) + support, math.ceil(max(ys)) + support
        return Region(int(left), int(top), int(right - left), int(bottom - top))

    @staticmethod
    def _reformat(src, src_display_w, src_display_h, target_w, target_h, p, src_box=None, dst_box=None):
        """Inverse-mapped resize/reposition into a `target_w`x`target_h` format, with sub-pixel
        filtering, mirroring `_transform`'s own structure (see its docstring). Placement is
        solved against `src_display_w`/`src_display_h` -- the input's own declared format -- not
        against `src`'s array extent, which may carry overscan (`src_box`, aligning the array to
        the source's real data window, exactly as `_transform` uses it)."""
        h, w = src.shape[:2]
        if src_box is None:
            src_box = Region(0, 0, w, h)
        if dst_box is None:
            dst_box = Region(0, 0, target_w, target_h)
        resize_type, center = p["resize_type"], p["center"]
        flip, flop, turn = p["flip"], p["flop"], p["turn"]
        scale_x, scale_y, offset_x, offset_y, _tw, _th = Evaluator._reformat_place(
            target_w, target_h, src_display_w, src_display_h, resize_type, center, turn)
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float32) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float32) + dst_box.y + 0.5)
        # Undo flip/flop/turn, in reverse of the order `_reformat_transformed_window` applies them,
        # then undo the placement scale/offset to land on a source sample coordinate.
        y_a = (target_h - gy) if flip else gy
        x_b = (target_w - gx) if flop else gx
        px, py = (y_a, x_b) if turn else (x_b, y_a)
        sx = (px - offset_x) / scale_x if scale_x else px
        sy = (py - offset_y) / scale_y if scale_y else py
        sx_frac = sx - 0.5 - src_box.x
        sy_frac = sy - 0.5 - src_box.y
        return Evaluator._resample(src, sx_frac, sy_frac, p["filter"]).astype(np.float32)

    @staticmethod
    def _cornerpin(src, p, src_box=None, dst_box=None):
        """Inverse-mapped projective (four-point) warp, sharing `_transform`'s own inverse-map-
        then-`_resample` structure. `_cornerpin_forward_matrix` is the one place the homography
        is solved, so the kernel and `_cornerpin_window` cannot disagree about which quad is the
        pre-warp side."""
        h, w = src.shape[:2]
        if src_box is None:
            src_box = Region(0, 0, w, h)
        if dst_box is None:
            dst_box = src_box
        forward = Evaluator._cornerpin_forward_matrix(p)
        inverse = np.linalg.inv(forward)
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float64) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float64) + dst_box.y + 0.5)
        denom = inverse[2, 0] * gx + inverse[2, 1] * gy + inverse[2, 2]
        denom = np.where(np.abs(denom) > 1e-9, denom, 1e-9)
        sx = (inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2]) / denom
        sy = (inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2]) / denom
        sx_frac = (sx - 0.5 - src_box.x).astype(np.float32)
        sy_frac = (sy - 0.5 - src_box.y).astype(np.float32)
        return Evaluator._resample(src, sx_frac, sy_frac, p["filter"]).astype(np.float32)

    @staticmethod
    def _transform(src, translate_x, translate_y, rotate, scale, center_x, center_y, filter,
                   src_box=None, dst_box=None, params=None):
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
        extras = params or {}
        if (translate_x == 0 and translate_y == 0 and (rotate % 360.0) == 0
                and scale == 1 and center_x == 0 and center_y == 0 and filter == "nearest"
                and not any(extras.get(name, default) != default for name, default in
                            (("skew_x", 0.0), ("skew_y", 0.0), ("invert", 0),
                             ("clamp", 0), ("black_outside", 0)))
                and extras.get("scale_mode", "uniform") == "uniform"
                and dst_box == src_box):
            return src.copy()
        p = dict(params or {}, translate_x=translate_x, translate_y=translate_y,
                 rotate=rotate, scale=scale, center_x=center_x, center_y=center_y)
        forward = Evaluator._transform_forward_matrix(p)
        inverse = np.linalg.inv(forward)
        # Destination pixel centres (world space). Nuke convention: integer pixel indices span
        # [i, i+1) and the centre sits at i + 0.5, which makes translate=0 sample the source on its
        # own pixel centres for the identity transform.
        gx, gy = np.meshgrid(np.arange(dst_box.width, dtype=np.float32) + dst_box.x + 0.5,
                             np.arange(dst_box.height, dtype=np.float32) + dst_box.y + 0.5)
        sx = inverse[0, 0] * gx + inverse[0, 1] * gy + inverse[0, 2]
        sy = inverse[1, 0] * gx + inverse[1, 1] * gy + inverse[1, 2]
        # Convert world sample coord to fractional pixel index inside the source array, which may
        # itself start away from the origin when the source carries overscan.
        sx_frac = sx - 0.5 - src_box.x
        sy_frac = sy - 0.5 - src_box.y
        clamp = bool(p.get("clamp", 0))
        result = Evaluator._resample(src, sx_frac, sy_frac, filter, clamp=clamp).astype(np.float32)
        if p.get("black_outside", 0):
            outside = ((sx_frac < -0.5) | (sx_frac > src.shape[1] - 0.5) |
                       (sy_frac < -0.5) | (sy_frac > src.shape[0] - 0.5))
            result[outside] = (0.0, 0.0, 0.0, 1.0)
        return result

    @staticmethod
    def _resample(src, sx_frac, sy_frac, filter, clamp=False):
        h, w = src.shape[:2]
        if clamp:
            sx_frac = np.clip(sx_frac, 0.0, max(0.0, w - 1.0))
            sy_frac = np.clip(sy_frac, 0.0, max(0.0, h - 1.0))
        if filter == "nearest":
            xi = np.floor(sx_frac + 0.5).astype(np.int32)
            yi = np.floor(sy_frac + 0.5).astype(np.int32)
            valid = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
            xi_c = np.clip(xi, 0, w - 1)
            yi_c = np.clip(yi, 0, h - 1)
            sampled = src[yi_c, xi_c]
            return np.where(valid[..., None], sampled, np.float32(0.0)).astype(np.float32)
        if filter == "bilinear":
            # Profiled at 4K (docs/BENCHMARKS-v0.33-m2.md's throughput section): the original four
            # independent `fetch(xi, yi)` closures each reclipped and revalidated `x0`/`x1`/`y0`/
            # `y1` from scratch, even though every one of those four arrays is reused by two of
            # the four corners. Clipping and validity are computed once per axis value here and
            # reused across the corners that share it; the four gathers and the weighted sum are
            # unchanged, so results are bit for bit identical to before.
            x0 = np.floor(sx_frac).astype(np.int32)
            y0 = np.floor(sy_frac).astype(np.int32)
            x1, y1 = x0 + 1, y0 + 1
            fx = (sx_frac - x0).astype(np.float32)
            fy = (sy_frac - y0).astype(np.float32)
            xc0, xc1 = np.clip(x0, 0, w - 1), np.clip(x1, 0, w - 1)
            yc0, yc1 = np.clip(y0, 0, h - 1), np.clip(y1, 0, h - 1)
            if clamp:
                p00, p10, p01, p11 = src[yc0, xc0], src[yc0, xc1], src[yc1, xc0], src[yc1, xc1]
            else:
                valid_x0, valid_x1 = (x0 >= 0) & (x0 < w), (x1 >= 0) & (x1 < w)
                valid_y0, valid_y1 = (y0 >= 0) & (y0 < h), (y1 >= 0) & (y1 < h)
                zero = np.float32(0.0)
                p00 = np.where((valid_x0 & valid_y0)[..., None], src[yc0, xc0], zero)
                p10 = np.where((valid_x1 & valid_y0)[..., None], src[yc0, xc1], zero)
                p01 = np.where((valid_x0 & valid_y1)[..., None], src[yc1, xc0], zero)
                p11 = np.where((valid_x1 & valid_y1)[..., None], src[yc1, xc1], zero)
            wx0, wx1 = (1 - fx)[..., None], fx[..., None]
            wy0, wy1 = (1 - fy)[..., None], fy[..., None]
            return (p00 * wx0 * wy0 + p10 * wx1 * wy0
                    + p01 * wx0 * wy1 + p11 * wx1 * wy1).astype(np.float32)
        if filter == "cubic":
            # Catmull-Rom (B=0, C=0.5): classic image-processing bicubic, sharper than Mitchell.
            a = -0.5
            def weight(t):
                at = np.abs(t)
                at2, at3 = at * at, at * at * at
                return np.where(at <= 1, (a + 2) * at3 - (a + 3) * at2 + 1,
                                a * at3 - 5 * a * at2 + 8 * a * at - 4 * a).astype(np.float32)
            shape, channels = sx_frac.shape, src.shape[2]
            sx, sy = sx_frac.reshape(-1), sy_frac.reshape(-1)
            if not sx.size:
                return np.zeros(shape + (channels,), dtype=np.float32)
            x_floor, y_floor = np.floor(sx), np.floor(sy)
            # Sixteen separate 2-D gathers over the whole source, each with its own clipping and mask, were most
            # of the cost. Instead: cut the source down to the window the taps can reach, pad it once (zeros for
            # black outside, the edge pixel for clamp) so that no tap needs a bounds test, and gather from the
            # flattened copy by linear offset, a band of pixels at a time so the working set stays in cache.
            # The weights and the order of the sums are the old ones, so the pixels are bit for bit the same.
            pad = 4
            bounds = [float(x_floor.min()), float(x_floor.max()), float(y_floor.min()), float(y_floor.max())]
            if np.all(np.isfinite(bounds)):
                x_lo, x_hi = int(min(max(bounds[0] - 1, 0), w - 1)), int(min(max(bounds[1] + 2, 0), w - 1))
                y_lo, y_hi = int(min(max(bounds[2] - 1, 0), h - 1)), int(min(max(bounds[3] + 2, 0), h - 1))
            else:
                x_lo, x_hi, y_lo, y_hi = 0, w - 1, 0, h - 1
            window = src[y_lo:y_hi + 1, x_lo:x_hi + 1]
            window_h, window_w = window.shape[:2]
            padded = np.pad(window, ((pad, pad), (pad, pad), (0, 0)), mode="edge" if clamp else "constant")
            stride = window_w + 2 * pad
            flat = np.ascontiguousarray(padded, dtype=np.float32).reshape(-1, channels)
            result = np.empty((sx.size, channels), dtype=np.float32)
            band = 65536
            for start in range(0, sx.size, band):
                stop = min(sx.size, start + band)
                x0, y0 = x_floor[start:stop], y_floor[start:stop]
                wx = np.stack([weight(f) for f in _cubic_taps((sx[start:stop] - x0).astype(np.float32))], axis=-1)
                wy = np.stack([weight(f) for f in _cubic_taps((sy[start:stop] - y0).astype(np.float32))], axis=-1)
                # A sample more than three pixels outside reads only padding, wherever it is.
                xi = np.clip(x0, x_lo - 3, x_lo + window_w + 1).astype(np.int64) - x_lo
                yi = np.clip(y0, y_lo - 3, y_lo + window_h + 1).astype(np.int64) - y_lo
                base = (yi + pad - 1) * stride + (xi + pad - 1)
                total = np.zeros((stop - start, channels), dtype=np.float32)
                for dy in range(4):
                    row = np.zeros((stop - start, channels), dtype=np.float32)
                    for dx in range(4):
                        taps = flat.take(base + (dy * stride + dx), axis=0)
                        taps *= wx[:, dx:dx + 1]
                        row += taps
                    row *= wy[:, dy:dy + 1]
                    total += row
                result[start:stop] = total
            return result.reshape(shape + (channels,))
        raise ValueError(f"Unknown transform filter: {filter}")
