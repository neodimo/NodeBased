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

* Filter shape (`shape`) "dissolves the filter shape between Gaussian at 0 and disc at 1": the weight of a sample inside the filter is
  shape * 1 + (1 - shape) * a Gaussian of sigma = size / 2, cut at the filter's edge. Filter aspect (`aspect`): "values less than 1 squeeze the
  filter on the x axis, and values larger than 1 squeeze it on the y axis" (the disc and the bladed filter). Catadioptric (`catadioptric`,
  `catadioptric_size`): the middle of the filter, out to that fraction of its size, is empty ("donut-shaped bokeh").
* Bloom (`bloom`, `bloom_threshold`, `bloom_gain`) and gamma correction (`bloom_gamma`): "highlights above the bloom threshold are multiplied by
  the bloom gain", "a gamma curve of 2.2 is applied before blurring and then reversed for the final output"; the picture is gamma-encoded first,
  then boosted, blurred, and decoded.
  The default of `shape` is 1 (a disc), the value Nuke's own node starts with and the video guide calls "a sharp disc"; the page's table lists 0.

* The bladed filter (lane 9 step S). The page states each knob's effect and no formula, so the construction is ours. The outer shape is the
  polygon of `blades` sides whose corners lie on the circle of the filter's size (`rotation` turns it counter-clockwise; the page's second
  `aspect ratio` knob, `aspect#2`, is the same control as the disc filter's, shown for the bladed type: one `aspect` here). `roundness` "controls the rounding of the filter polygon's sides, where
  zero is equal to no rounding": each side bows out to the corner circle, the edge at an angle moving from the straight side by that
  fraction of the way to the circle, so 0 is the straight polygon, 1 the disc. `inner_size` is "the size of the inner polygon, as a
  percentage of the outer polygon" (the same polygon scaled about the centre); inside it the filter's weight is `inner_brightness` ("0 is
  equal to black and 1 to white"), outside it, out to the rim, 1. `inner_feather`: "with values larger than 0.5 your feather effect is
  outward and, respectively, if smaller than 0.5, the feather effect is inward. A value of 0.5 produces no feathering": a linear ramp
  between the inner brightness and 1, starting at the inner polygon's edge and ending |2 * feather - 1| of the way to the rim (outward) or
  of the way to the centre (inward). Bladed shapes carry weights now, so a filter with a dim middle blurs a point into a ring.
* The image filter (the wired kernel; same step). `filter_channel` names the plane read as the convolution matrix (the alpha by default);
  `use_input_channels` reads the filter's own red, green and blue for the image's red, green and blue (a coloured filter fringes the
  bokeh). The filter is resized to the blur size with `image_filter` ("remapped pixels carry their original values" for Impulse, more
  smoothing for Cubic, Mitchell and Parzen, sharpening for Keys, Simon and Rifman; `clamp_image_filter` "corrects the haloing" of the
  sharpening ones, so the resized weights stay inside the range of the filter's own). With `legacy_resize_mode` on, the old resize is used
  (nearest pixels) and `filter_bounds` says what is resized: `shape` the box around the filter's non-zero pixels (its centre is the filter's
  centre), `format` the whole filter picture, so the filter image can be offset inside it. Nuke's Notch is not in NodeBased's filter list.
  A bright point becomes the filter as the filter is drawn ("any out-of-focus highlights assume the shape of the filter image"): the weights are
  gathered at the sample's offset from the pixel, which turns an asymmetric filter half way round, so the new filters are turned back; a
  node saved before these (`legacy_zdefocus_filters` missing or 1) keeps the straight polygon, the unweighted mean, the nearest resize and
  the filter image the way round it was.

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
       "focal_y": POINT_DEFAULT[1], "blur_dof": 1, "autoLayerSpacing": 1, "nLayers": 20, "layerCurve": 1.0,
       "shape": 0.0, "aspect": 1.0, "catadioptric": 0, "catadioptric_size": 0.3, "bloom": 0, "bloom_threshold": 0.8, "bloom_gain": 2.0, "bloom_gamma": 0,
       "legacy_zdefocus_filters": 0, "roundness": 0.2, "inner_size": 0.8, "inner_feather": 1.0, "inner_brightness": 0.8,
       "legacy_resize_mode": 0, "filter_channel": "rgba.alpha", "use_input_channels": 0, "filter_bounds": "shape", "image_filter": "cubic",
       "clamp_image_filter": 0}
UPGRADE = {**NEW, "shape": 1.0, "legacy_zdefocus_focus": 1, "legacy_zdefocus_filters": 1}
LIMITS = {"legacy_zdefocus_focus": (0, 1), "show_image": (0, 1), "fill_foreground": (0, 1), "focal_x": (-100000.0, 100000.0),
          "focal_y": (-100000.0, 100000.0), "blur_dof": (0, 1), "autoLayerSpacing": (0, 1), "nLayers": (1, MAX_LAYERS),
          "layerCurve": (-5.0, 5.0), "shape": (0.0, 1.0), "catadioptric": (0, 1), "catadioptric_size": (0.0, 1.0), "bloom": (0, 1),
          "bloom_threshold": (-1000.0, 1000.0), "bloom_gain": (0.0, 1000.0), "bloom_gamma": (0, 1),
          "legacy_zdefocus_filters": (0, 1), "roundness": (0.0, 1.0), "inner_size": (0.0, 1.0), "inner_feather": (0.0, 1.0),
          "inner_brightness": (0.0, 1.0), "legacy_resize_mode": (0, 1), "use_input_channels": (0, 1), "clamp_image_filter": (0, 1)}
CHOICES = {"zdefocus_output": list(OUTPUTS)}


def needs_full_frame(p):
    """A focal point reads the depth at one place of the whole picture and the filter shape view is centred on it: tiles cannot do either."""
    if p.get("bokeh_shape") == "image":
        return True                                       # the filter image is a whole picture the filter is resized from, not a tile of it
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


GAMMA = 2.2


def encode(image, p):
    """Gamma correction, then the bloom boost, on the picture before it is blurred."""
    out = image.astype(np.float32)
    if int(p.get("bloom_gamma", 0)):
        out[..., :3] = np.power(np.maximum(out[..., :3], 0.0), 1.0 / GAMMA)
    if int(p.get("bloom", 0)):
        rgb = out[..., :3]
        out[..., :3] = np.where(rgb > float(p.get("bloom_threshold", 0.8)), rgb * float(p.get("bloom_gain", 2.0)), rgb)
    return out


def decode(image, p):
    """The gamma curve reversed on the blurred picture."""
    if not int(p.get("bloom_gamma", 0)):
        return image
    out = image.copy()
    out[..., :3] = np.power(np.maximum(out[..., :3], 0.0), GAMMA)
    return out


def shape_weights(disk, yy, xx, r, p):
    """Weights of the samples inside a disc of size r: shape dissolves a Gaussian (0) into the flat disc (1)."""
    mix = float(p.get("shape", 1.0))
    if mix >= 1.0 or r <= 0:
        return None
    sigma = max(r / 2.0, 1e-6)
    return disk * (mix + (1.0 - mix) * np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma)))


def new_filters(p):
    """True when the bladed and image filters follow the page (a node saved before them keeps the old ones)."""
    return not int(p.get("legacy_zdefocus_filters", 1))


def aspect_axes(p):
    """Divisors of (x, y) that squeeze the filter: aspect < 1 narrows x, aspect > 1 narrows y."""
    aspect = max(0.1, float(p.get("aspect", 1.0)))
    return min(1.0, aspect), min(1.0, 1.0 / aspect)


def hollow(disk, yy, xx, r, p):
    """Catadioptric: the middle of the filter, out to `catadioptric_size` of its size, is empty."""
    if not int(p.get("catadioptric", 0)) or r <= 0:
        return disk
    inner = float(p.get("catadioptric_size", 0.3)) * r
    return disk & (xx * xx + yy * yy >= inner * inner)


def bladed(yy, xx, r, p):
    """The bladed filter at size r on the (yy, xx) grid already squeezed by the aspect ratio: (inside mask, weights).

    The radial coordinate of a pixel is its distance over the rounded polygon's edge in its direction, so 1 is the rim and `inner_size` the
    inner polygon's edge; the weight is `inner_brightness` inside that, 1 beyond it, and a linear ramp across the feather."""
    blades = max(2, int(p.get("blade_count", 5)))
    # The grid is the sample offsets from the pixel being filled; the point it receives from lies at minus that offset, so the filter appears in
    # the picture as drawn: the angle is the picture's, counter-clockwise from the right, with y up ("positive values produce counter-clockwise rotation").
    theta = np.arctan2(yy, -xx) - math.radians(float(p.get("blade_rotation", 0.0)))
    sector = (theta + math.pi / blades) % (2 * math.pi / blades) - math.pi / blades
    straight = r * math.cos(math.pi / blades) / np.maximum(np.cos(sector), 1e-6)
    roundness = min(1.0, max(0.0, float(p.get("roundness", 0.2))))
    edge = (1.0 - roundness) * straight + roundness * r
    distance = np.sqrt(xx * xx + yy * yy)
    inside = distance <= edge + 1e-9
    level = distance / np.maximum(edge, 1e-9)
    size = min(1.0, max(0.0, float(p.get("inner_size", 0.8))))
    brightness = float(p.get("inner_brightness", 0.8))
    feather = min(1.0, max(0.0, float(p.get("inner_feather", 1.0))))
    spread = abs(2.0 * feather - 1.0)
    if feather >= 0.5:
        start, stop = size, size + spread * (1.0 - size)
    else:
        start, stop = size - spread * size, size
    if stop - start < 1e-9:
        ramp = np.where(level < size, 0.0, 1.0)
    else:
        ramp = np.clip((level - start) / (stop - start), 0.0, 1.0)
    weights = (brightness + (1.0 - brightness) * ramp) * inside
    return inside, weights.astype(np.float64)


def _filter_weight(name):
    """The weight function of the image filter `name` (the page's names, as NodeBased's filter list spells them)."""
    from . import transformbehaviour as tb
    name = str(name).lower()
    if name in tb.KERNELS:
        return tb.KERNELS[name][0]
    if name == "bilinear":
        return lambda x: np.maximum(0.0, 1.0 - np.abs(x))
    return tb._keys(0.0)                         # cubic: the smoothest member of the cubic family without negative lobes


def _taps(n_out, n_in, name):
    """The (n_out, n_in) matrix that resamples a line of n_in samples to n_out by pixel centres, widening the filter when it shrinks."""
    position = (np.arange(n_out) + 0.5) * n_in / n_out - 0.5          # pixel centres: the filter's outer edges stay its outer edges
    scale = max(1.0, n_in / n_out)
    matrix = _filter_weight(name)((position[:, None] - np.arange(n_in)[None, :]) / scale)
    total = matrix.sum(axis=1, keepdims=True)
    return matrix / np.where(np.abs(total) < 1e-12, 1.0, total)


def kernel_planes(kernel, p):
    """The wired filter picture as float64 planes (h, w, 1) or, with `use_input_channels`, (h, w, 4), cropped to the box round the filter's
    non-zero pixels (its bounding box: the filter's centre is the box's centre), except that the legacy resize with `format` bounds keeps
    the whole picture. A node saved before the controls takes the whole picture. `kernel` is a pixel array."""
    from . import ops2d_chrome as chrome
    pixels = np.asarray(kernel, np.float64)
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if int(p.get("use_input_channels", 0)) and pixels.shape[2] >= 4:
        planes = pixels[..., :4]
    else:
        channel = str(p.get("filter_channel", "rgba.alpha"))
        if pixels.shape[2] >= 4:
            planes = chrome.resolve_channel(channel, pixels.astype(np.float32))[..., :1].astype(np.float64)
        else:
            planes = pixels[..., :1]
    if new_filters(p) and not (int(p.get("legacy_resize_mode", 0)) and str(p.get("filter_bounds", "shape")) == "format"):
        ys, xs = np.nonzero(np.any(pixels != 0.0, axis=2))                 # the box round every channel's non-zero pixels: the image's bounds
        if len(ys):
            planes = planes[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    return planes


def image_grid(kernel, radius, p):
    """The wired filter resized to (2 * radius + 1) squares: (grid, inside mask). Planes are (n, n) or (n, n, 3)."""
    planes = kernel_planes(kernel, p)
    kh, kw = planes.shape[:2]
    n = 2 * radius + 1
    if not new_filters(p) or int(p.get("legacy_resize_mode", 0)) or str(p.get("image_filter", "cubic")).lower() in ("impulse", "nearest"):
        if new_filters(p) and not int(p.get("legacy_resize_mode", 0)):      # Impulse: each pixel of the resized filter carries an original value
            ky, kx = np.arange(n) * kh // n, np.arange(n) * kw // n
        else:
            ky = np.minimum(kh - 1, np.maximum(0, (np.arange(n) / max(1, 2 * radius) * (kh - 1)).astype(int)))
            kx = np.minimum(kw - 1, np.maximum(0, (np.arange(n) / max(1, 2 * radius) * (kw - 1)).astype(int)))
        grid = planes[np.ix_(ky, kx)]
    else:
        name = p.get("image_filter", "cubic")
        grid = np.einsum("ik,kjc->ijc", _taps(n, kh, name), planes)
        grid = np.einsum("jk,ikc->ijc", _taps(n, kw, name), grid)
        if int(p.get("clamp_image_filter", 0)):
            grid = np.clip(grid, planes.min(axis=(0, 1)), planes.max(axis=(0, 1)))
    if grid.ndim == 3 and grid.shape[2] == 1:
        grid = grid[..., 0]
    if new_filters(p):
        grid = grid[::-1, ::-1]                      # a gather sees the point at minus the offset, so the filter is turned round to appear as it is drawn
    if not new_filters(p):
        grid = grid.astype(np.float32)               # a saved node's filter, bit for bit
    inside = (grid != 0).any(axis=-1) if grid.ndim == 3 else (grid != 0)
    if not new_filters(p):
        inside = (grid > 0).any(axis=-1) if grid.ndim == 3 else (grid > 0)
    return grid, inside
