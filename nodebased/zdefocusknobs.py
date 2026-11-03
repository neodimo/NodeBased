"""ZDefocus's main tab (lane 9, 2D behaviour step M), as the Nuke 17.0 Reference Guide states it.

The page says what each control does but gives no formula for the blur, so the numbers below are ours; every sentence quoted is the guide's.

* Blur size (`size`, `max_size`): "the size of the blur at infinite depth. Blur nearer the camera than the focus plane may be larger".
  A thin lens does exactly that: at a distance d from a plane in focus at c the blur is size * |d - c| / d, which tends to `size` as d grows
  and exceeds it in front of the plane. The result is "clipped at maximum". (math `direct` and `-direct` keep the page's own rule: Z times size.)
* Focus plane and depth of field (`focal_plane`, `depth_of_field`): "a depth slice around the focus plane that is entirely in focus"; the slice
  is c - dof/2 .. c + dof/2. Outside it the distance is measured from the nearer edge of the slice. "Note that if depth of field is set to 0,
  nothing is displayed in green": a slice of nothing.
* Blur inside (`blur_dof`): "a small amount of blur is applied to the in-focus region ... a smoother transition". Inside the slice the radius
  rises from 0 at the focus plane to INSIDE_RADIUS pixels (ours) at the slice's edge; off, the slice is sharp.
* Focal point (`focal_x`, `focal_y`): "the position of the focal point widget in the Viewer. Adjusting the Viewer widget updates the focus plane".
  Here a point other than the page's default (200, 200) that lies inside the picture sets the focus plane to the distance read there (x, y in
  picture pixels from the top left, as the viewer reports a click); the default and any point outside the picture leave `focal_plane` alone.
* Fill foreground (`fill_foreground`): "Nuke attempts to compensate for missing information by filling regions in the foreground which are
  revealed when the foreground goes out of focus". A pixel nearer than the slice gathers only samples at its own depth or nearer; on, the
  samples it cannot take are filled by the ones it can (the mean of those); off, they count as empty, so a blurred foreground fades into
  nothing at its edge ("blurry artefacts", where on gives "sharp edge artefacts" at worst).
* Layers (`autoLayerSpacing`, `nLayers`, `layerCurve`): "ZDefocus splits the image up into layers, each ... processed with a single blur size".
  Here the blur sizes between 0 and maximum are the layers: n sizes at maximum * t ^ 2^curve, t = 0 .. 1 in n steps (curve 0 evenly spaced,
  "positive values concentrate the layers closer to the in-focus region", negative ones towards the maximum), at most 256. Automatic spacing
  "works out how many depth layers to use, based on the maximum blur size": quarter pixels to 2, half pixels to 4, whole pixels beyond (so
  whole-pixel blur sizes are layers of their own). A pixel takes the nearest layer's size.
* Output (`zdefocus_output`) and `show_image`: result; focal plane setup (red in front of the slice, green inside it, blue behind it);
  layer setup (the same colours after the sizes are snapped to layers: brightness is the layer number); filter shape setup (the filter in
  white, centred, at the maximum size). With show image on, a setup is half the picture plus half the colours; off, the colours alone.
  The page's red, green and blue outputs say nothing of what they show and are not offered.

A node saved before this page was followed keeps its picture (`legacy_zdefocus_focus`).
"""
from __future__ import annotations

import math

import numpy as np

INSIDE_RADIUS = 2.0
MAX_LAYERS = 256
OUTPUTS = ("result", "focal plane setup", "layer setup", "filter shape setup")
POINT_DEFAULT = (200.0, 200.0)
NEW = {"legacy_zdefocus_focus": 0, "zdefocus_output": "result", "show_image": 1, "fill_foreground": 1, "focal_x": POINT_DEFAULT[0],
       "focal_y": POINT_DEFAULT[1], "blur_dof": 1, "autoLayerSpacing": 1, "nLayers": 20, "layerCurve": 1.0}
UPGRADE = {**NEW, "legacy_zdefocus_focus": 1}
LIMITS = {"legacy_zdefocus_focus": (0, 1), "show_image": (0, 1), "fill_foreground": (0, 1), "focal_x": (-100000.0, 100000.0),
          "focal_y": (-100000.0, 100000.0), "blur_dof": (0, 1), "autoLayerSpacing": (0, 1), "nLayers": (1, MAX_LAYERS),
          "layerCurve": (-5.0, 5.0)}
CHOICES = {"zdefocus_output": list(OUTPUTS)}


def needs_full_frame(p):
    """A focal point reads the depth at one place of the whole picture and the filter shape view is centred on it: tiles cannot do either."""
    if int(p.get("legacy_zdefocus", 1)) or int(p.get("legacy_zdefocus_focus", 1)):
        return False
    return (float(p.get("focal_x", POINT_DEFAULT[0])), float(p.get("focal_y", POINT_DEFAULT[1]))) != POINT_DEFAULT \
        or p.get("zdefocus_output", "result") == "filter shape setup"


def layer_sizes(p):
    """The blur sizes (pixels) the layers stand for, smallest first, between 0 and `max_size`."""
    maximum = abs(float(p.get("max_size", 20.0)))
    if int(p.get("autoLayerSpacing", 1)):
        # Close together near the focus plane, wider far away: quarter pixels to 2, half pixels to 4, whole pixels beyond, and the maximum.
        steps = list(np.arange(0.0, 2.0, 0.25)) + list(np.arange(2.0, 4.0, 0.5)) + list(np.arange(4.0, maximum, 1.0))
        sizes = np.array(sorted({float(v) for v in steps if v < maximum} | {maximum}), np.float64)
        if len(sizes) > MAX_LAYERS:
            sizes = np.linspace(0.0, maximum, MAX_LAYERS)
        return sizes
    count, curve = max(2, min(MAX_LAYERS, int(p.get("nLayers", 20)))), float(p.get("layerCurve", 1.0))
    return maximum * np.linspace(0.0, 1.0, count) ** (2.0 ** curve)


def snap_to_layers(radii, p):
    """Each radius moved to the nearest layer's size; the layer number beside it."""
    sizes = layer_sizes(p)
    index = np.abs(np.asarray(radii, np.float64)[..., None] - sizes).argmin(axis=-1)
    return sizes[index].astype(np.float32), index


def focus(distance, p):
    """(focus plane, half the depth of field): the plane is read from the depth at the focal point when that point is set and inside."""
    centre = float(p.get("focal_plane", 1.0))
    half = max(0.0, float(p.get("depth_of_field", 1.0))) / 2.0
    fx, fy = float(p.get("focal_x", POINT_DEFAULT[0])), float(p.get("focal_y", POINT_DEFAULT[1]))
    h, w = distance.shape
    if (fx, fy) != POINT_DEFAULT and 0 <= int(math.floor(fx)) < w and 0 <= int(math.floor(fy)) < h:
        centre = float(distance[int(math.floor(fy)), int(math.floor(fx))])
    return centre, half


def blur_sizes(distance, centre, half, p):
    """The blur radius of each pixel by the thin-lens rule above (the older modes `direct` and `-direct` do not come here)."""
    nearest = np.clip(distance, centre - half, centre + half)
    out = float(p.get("size", 25.0)) * np.abs(distance - nearest) / np.maximum(np.abs(distance), 1e-6)
    if int(p.get("blur_dof", 1)) and half > 0:
        inside = np.abs(distance - centre) <= half
        out = np.where(inside, INSIDE_RADIUS * np.abs(distance - centre) / half, out)
    return np.minimum(out, abs(float(p.get("max_size", 20.0)))).astype(np.float32)


def side(distance, centre, half):
    """-1 in front of the depth-of-field slice, 0 inside it, +1 behind it. A depth of field of 0 is a slice of nothing: no green."""
    if half <= 0:
        return np.where(distance < centre, -1, 1)
    return np.where(distance < centre - half, -1, np.where(distance > centre + half, 1, 0))


def setup_colours(sides, level=None):
    """Red in front, green inside, blue behind; `level` (0 .. 1) scales red and blue (the layer number)."""
    scale = np.ones(sides.shape, np.float32) if level is None else (0.25 + 0.75 * level).astype(np.float32)
    inside = sides == 0
    return np.stack([(sides < 0) * scale, inside.astype(np.float32), (sides > 0) * scale], axis=-1).astype(np.float32)


def setup_view(image, colours, p):
    """The setup picture: half the picture plus half the colours with show image on, the colours alone (alpha 1) with it off."""
    if int(p.get("show_image", 1)):
        out = image.copy()
        out[..., :3] = 0.5 * image[..., :3] + 0.5 * colours
        return out.astype(np.float32)
    out = np.ones(image.shape, np.float32)
    out[..., :3] = colours
    return out


def filter_shape_view(image, mask, p):
    """The filter drawn in white on black at the maximum size, centred (the disc/bladed/image choice is passed in as `mask`)."""
    h, w = image.shape[:2]
    out = np.zeros(image.shape, np.float32)
    out[..., 3] = 1.0
    ry, rx = mask.shape
    top, left = h // 2 - ry // 2, w // 2 - rx // 2
    for y in range(ry):
        for x in range(rx):
            if 0 <= top + y < h and 0 <= left + x < w and mask[y, x]:
                out[top + y, left + x, :3] = 1.0
    return out
