"""Behaviour fixes to the Draw nodes (lane 9, 2D behaviour step C): Rectangle, Radial, Ramp, Noise and Grid, plus the shared ramp.

Each fix keeps the older reading behind a `legacy_*` parameter. A parameter dict that predates the knob reads as an older document's
node (a missing flag is the legacy value), and `upgrade_params` gives a saved node the explicit legacy value on load, so an older
document renders bit for bit as it did. A node created now gets the new reading from its SPECS defaults.

* Axes (`legacy_draw_axes`, Rectangle, Radial, Ramp, Noise, Grid): Nuke counts y upwards. The Draw pages say the ramp's point 0 end is
  "by default, the bottom end" and its point 1 end "the top end" (point 0 100, 100 and point 1 100, 400). NodeBased draws row 0 at
  the top, so the vertical coordinate of every position knob (the area's y, the ramp's points, the grid's offset, the noise's translate
  and centre) now counts rows from the bottom of the format. Older nodes counted them from the top.
* Ramp sampling (the same flag): a colour ramp is sampled at the pixel's centre, a half pixel in from its index, which is where Nuke
  measures its coordinates (the Expression node's x and y are pixel centres, as 2D behaviour lane 2 found). Older nodes sampled at the
  index.
* Ramp type (`ramp_type`, Ramp): the guide's list is linear, plinear, smooth0, smooth1 and smooth; the older Ramp was only linear.
* Noise (the same flag): the noise is sampled in 3D, so `zoffset` moves through the volume smoothly (it used to step between whole slices),
  `xrotate` and `yrotate` turn the volume about its x and y axes (they used to be added to the in-plane rotation), `size` is an x and a
  y size ("low y values relative to x produces horizontal noise"), the transform is scale, skew, rotate and translate about the centre
  with y up, and the noise is the shape's coverage coloured by `color` (and the ramp), so a Noise draws over its input like the other
  Draw nodes. An older Noise sampled a 2D slice and drew opaque grey.
* Grid (the same flag): a Grid's `number` defaults to Nuke's 10 squares, and the Color tab's ramp is drawn.
* Dither (`legacy_dither`): the page says "up to 0.5 of this amount is added or subtracted from each pixel", `monodither` uses one pattern
  for every channel and `static seed` keeps the pattern from changing with the frame. The older node rounded the picture to `bits` bits
  after adding triangular noise of up to a bit; `bits` is now an optional rounding after the dither (0, the default of a new node, is none).
* Grain (`legacy_grain`): the intensities, irregularity and black point of Nuke's Grain, its minimum rgb and `Apply only through alpha`.
  The grain's amplitude follows the pixel's brightness, from the `black` amount at black to the full intensity at white (an older node did
  that only when `luminance_weighted` was on); `irregularity` bends the grain's lattice so the pattern is not a regular grid; the node's
  `minimum` raises any value below it; and with `Apply only through alpha` on (the page's default) the grain is scaled by the picture's alpha.
  Foundry does not publish the grain's amplitude: an intensity of 1 is `GRAIN_SCALE` in the picture's units, set so Nuke's default red
  intensity gives the amplitude the older node's default had.
* Radial (`plinear`, `ramp`, `p0`, `p1`, `color0`): "perceptually linear" and the colour tab the Radial page lists next to Rectangle's.
  Older Radial nodes get the ramp off and plinear off, so they draw as before.

* Defocus (`legacy_defocus`, `scale`): the page says `aspect ratio` "adjusts the width of the defocus disc without altering its height" and
  `scaling` "scales the defocus disc simultaneously on the x and y axes" (it can be split into a width and a height). The older node
  kept the width and shrank the height by the aspect. `defocus` stays the disc's radius in pixels (the page says only "the size").
* ZDefocus (`legacy_zdefocus`, `zdefocus_math`, `size`): the page's `math` choices turn the depth channel into a distance, and `direct` makes
  "the Z value in the depth channel directly controls blur: if Z is 0.5, then the blur size will be 0.5 times the value of the size
  control (unless this is bigger than maximum, in which case it will be clamped to maximum)". `-direct` blurs by the same amount with
  the layers in the opposite order. depth, -depth, far = 0 and far = -0 convert the channel to a distance by the page's formulas (Z,
  -Z, 1/Z and -1/Z); the page names far = 1 and far = -1 without their near and far planes, so they are not offered. The older node had
  only depth and 1/depth.
* EdgeExtend (`legacy_edge_extend`, `source_premultiplied`, `premultiply`, `erode`): the page's Source Is Premultiplied (off by default:
  "no premultiplication is assumed"), Premultiply (on by default: the output is premultiplied) and Erode ("negative values dilate the
  matte and positive values erode the matte"). The older node read its input as premultiplied and wrote straight colour.

* Roto (`legacy_roto_alpha`): the page's `replace` ("existing channels are cleared to black before drawing into them ... the incoming image
  already has an alpha channel which you want to throw away") says that with replace off the shapes are drawn into the incoming alpha:
  where they do not reach it stays, and where they do the matte goes over it. The older node wrote the matte over the plate's alpha.

* RotoPaint (`legacy_paint_replace`): the same `replace` ("existing channels are cleared to black before drawing into them") clears the image
  the strokes are drawn on; the plate still feeds clone and background-source strokes. The older node ignored the knob.

Pure NumPy, so the evaluator, the tile executor and the tests read one arithmetic.
"""
from __future__ import annotations

import numpy as np

from . import zdefocusknobs as _zdefocusknobs

FLAG = "legacy_draw_axes"
KINDS = ("Rectangle", "Radial", "Ramp", "Noise", "Grid")
NOISE_COLOUR = {"red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0}
RAMP_TYPES = ("linear", "plinear", "smooth0", "smooth1", "smooth")
# The exponent a "perceptually linear" ramp uses: the value a display gamma of about 2.2 shows as steps that look even.
PLINEAR_GAMMA = 2.2
RAMP_POINTS = {"p0_x": 100.0, "p0_y": 100.0, "p1_x": 100.0, "p1_y": 400.0,
               "color0_red": 0.0, "color0_green": 0.0, "color0_blue": 0.0, "color0_alpha": 1.0}

NEW = {kind: {FLAG: 0} for kind in KINDS}
NEW["Ramp"] = {FLAG: 0, "ramp_type": "linear"}
NEW["Radial"] = {FLAG: 0, "plinear": 0, "ramp": "none", **RAMP_POINTS}
NEW["Grid"] = {FLAG: 0, "ramp": "none", **RAMP_POINTS}
NEW["Noise"] = {FLAG: 0, **NOISE_COLOUR}
NEW["Dither"] = {"legacy_dither": 0, "monodither": 0, "static_seed": 0, "bits": 0, "dither_amount": 0.1}
NEW["Grain"] = {"legacy_grain": 0, "red_i": 0.6, "green_i": 0.6, "blue_i": 0.6, "minimum_red": 0.0, "minimum_green": 0.0, "minimum_blue": 0.0,
                "maskgrain": 1, "invert_mask": 0, "red_intensity": 0.416, "green_intensity": 0.46, "blue_intensity": 0.85}
NEW["Defocus"] = {"legacy_defocus": 0, "scale": 1.0}
NEW["ZDefocus"] = {"legacy_zdefocus": 0, "zdefocus_math": "depth", "size": 25.0, **_zdefocusknobs.NEW}
NEW["EdgeExtend"] = {"legacy_edge_extend": 0, "source_premultiplied": 0, "premultiply": 1, "erode": 0.0}
NEW["Roto"] = {"legacy_roto_alpha": 0, "legacy_roto_shapes": 0}
NEW["LightWrap"] = {"legacy_lightwrap_grade": 0, "legacy_lightwrap_luma": 0}
NEW["RotoPaint"] = {"legacy_paint_replace": 0}
LEGACY = {kind: {FLAG: 1} for kind in KINDS}
LEGACY["Roto"] = {"legacy_roto_alpha": 1, "legacy_roto_shapes": 1}
LEGACY["LightWrap"] = {"legacy_lightwrap_grade": 1, "legacy_lightwrap_luma": 1}
LEGACY["RotoPaint"] = {"legacy_paint_replace": 1}
LEGACY["Defocus"] = {"legacy_defocus": 1, "scale": 1.0}
LEGACY["ZDefocus"] = {"legacy_zdefocus": 1, "zdefocus_math": "depth", "size": 25.0, **_zdefocusknobs.UPGRADE}
LEGACY["EdgeExtend"] = {"legacy_edge_extend": 1, "source_premultiplied": 1, "premultiply": 0, "erode": 0.0}
ZDEFOCUS_MATH = ("direct", "depth", "far = 0", "-direct", "-depth", "far = -0")
LEGACY["Ramp"] = {FLAG: 1, "ramp_type": "linear"}
LEGACY["Radial"] = {FLAG: 1, "plinear": 0, "ramp": "none", **RAMP_POINTS}
LEGACY["Grid"] = {FLAG: 1, "ramp": "none", **RAMP_POINTS}
LEGACY["Noise"] = {FLAG: 1, **NOISE_COLOUR}
LEGACY["Dither"] = {"legacy_dither": 1, "monodither": 0, "static_seed": 0}
LEGACY["Grain"] = {"legacy_grain": 1, "red_i": 0.6, "green_i": 0.6, "blue_i": 0.6, "minimum_red": 0.0, "minimum_green": 0.0, "minimum_blue": 0.0,
                   "maskgrain": 0, "invert_mask": 0}
LIMITS = {"legacy_roto_alpha": (0, 1), "legacy_roto_shapes": (0, 1), "legacy_lightwrap_grade": (0, 1), "legacy_lightwrap_luma": (0, 1), "legacy_paint_replace": (0, 1), "legacy_defocus": (0, 1), "legacy_zdefocus": (0, 1), "legacy_edge_extend": (0, 1), "source_premultiplied": (0, 1), "premultiply": (0, 1),
          "erode": (-100.0, 100.0), FLAG: (0, 1), "plinear": (0, 1), "legacy_dither": (0, 1), "monodither": (0, 1), "static_seed": (0, 1), "legacy_grain": (0, 1),
          "red_i": (0.0, 1.0), "green_i": (0.0, 1.0), "blue_i": (0.0, 1.0), "minimum_red": (-1000.0, 1000.0), "minimum_green": (-1000.0, 1000.0),
          "minimum_blue": (-1000.0, 1000.0), "maskgrain": (0, 1), **_zdefocusknobs.LIMITS}
GRAIN_SCALE = 0.12


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What an older document's node of `kind` gets on load."""
    return dict(LEGACY.get(kind, {}))


def roto_alpha(coverage, plate_alpha, p):
    """The alpha channel of a Roto drawn over a wired background: the matte over the plate's own alpha (or, for an older node, the matte
    alone). Shared by the evaluator and the tile path."""
    if int(p.get("legacy_roto_alpha", 1)):
        return coverage
    return coverage + plate_alpha * (np.float32(1.0) - coverage)


def paint_replaces(p):
    """True when a RotoPaint draws its strokes on a cleared canvas (the guide's Replace: "existing channels are cleared to black before drawing
    into them"). An older node ignored the knob."""
    return bool(int(p.get("replace", 0))) and not int(p.get("legacy_paint_replace", 1))


def roto_premultiply_alpha(alpha, coverage, p):
    """premultiply rgba on a Roto: "premultiplies the alpha against itself (a*a)"; the older node multiplied the matte by the coverage."""
    if int(p.get("legacy_roto_alpha", 1)):
        return alpha * coverage
    return alpha * alpha


def is_legacy(p):
    """True when `p` reads as an older document's draw node (a missing flag is an older node)."""
    return bool(int(p.get(FLAG, 1)))


COVERAGE_KINDS = ("Rectangle", "Radial", "Ramp", "Grid", "Noise")


def inverts_coverage(kind, p):
    """True when the node's `invert` is applied to the shape's coverage, before the colour and its alpha: the guide's "inverts the new
    rectangle and non-rectangle areas". An older node inverted the finished alpha instead (a colour with alpha 0.8 then gave 0.2 inside
    and 1 outside)."""
    return kind in COVERAGE_KINDS and not is_legacy(p)


def cover(p, coverage):
    """`coverage` (0..1) inverted when the node's invert is on and follows Nuke."""
    if not is_legacy(p) and int(p.get("invert", 0)):
        return 1.0 - coverage
    return coverage


def rows(p, y0, h):
    """The vertical coordinate of each of `h` rows starting at absolute row `y0`: the row index in an older node, the index counted
    from the bottom of the format (`height - 1 - row`) in a node that follows Nuke."""
    index = np.arange(y0, y0 + h, dtype=np.float64)
    return index if is_legacy(p) else float(p["height"]) - 1.0 - index


def ease(t, kind):
    """The ramp's amount for a position `t` along it (0 at point 0, 1 at point 1, already clamped): the guide's five ramp types.

    `smooth0` eases into point 0 (the colours there are spread wider), `smooth1` into point 1 and `smooth` into both; the guide
    gives the words and not the curves, so these are the usual quadratic and smoothstep forms. `plinear` is the linear ramp seen
    through a display gamma."""
    if kind == "smooth0":
        return t * t
    if kind == "smooth1":
        return 1.0 - (1.0 - t) * (1.0 - t)
    if kind == "smooth":
        return t * t * (3.0 - 2.0 * t)
    if kind == "plinear":
        return t ** PLINEAR_GAMMA
    return t


def ramp_position(p, x0, y0, w, h):
    """Where along the ramp from point 0 to point 1 each pixel sits (unclamped, shape (h, w)). A node that follows Nuke measures
    from the pixel's centre and from the bottom of the format; an older one from the pixel's index and the top."""
    legacy = is_legacy(p)
    xs = np.arange(x0, x0 + w, dtype=np.float64)
    ys = rows(p, y0, h)
    if not legacy:
        xs, ys = xs + 0.5, ys + 0.5
    xx, yy = np.meshgrid(xs, ys)
    p0x, p0y = float(p.get("p0_x", 0.0)), float(p.get("p0_y", 0.0))
    dx, dy = float(p.get("p1_x", 1.0)) - p0x, float(p.get("p1_y", 0.0)) - p0y
    denom = dx * dx + dy * dy
    if denom < 1e-12:
        return np.zeros_like(xx)
    return ((xx - p0x) * dx + (yy - p0y) * dy) / denom


def ramp_weight(p, x0, y0, w, h, kind):
    """The 0..1 weight of the high colour at each pixel for the ramp `kind` of the node's p0 and p1."""
    t = np.clip(ramp_position(p, x0, y0, w, h), 0.0, 1.0)
    return ease(t, kind)


def ramp_colour(p, x0, y0, w, h, kind, high):
    """Straight RGBA across the area: `color0_*` at point 0 to `high` (r, g, b, a) at point 1."""
    weight = ramp_weight(p, x0, y0, w, h, kind)[..., None]
    low = np.array([p.get("color0_red", 0.0), p.get("color0_green", 0.0), p.get("color0_blue", 0.0), p.get("color0_alpha", 1.0)],
                   dtype=np.float64)
    return low[None, None, :] * (1.0 - weight) + np.asarray(high, dtype=np.float64)[None, None, :] * weight


# ------------------------------------------------------------------------------------------------------------------ Noise
def _hash3(ix, iy, iz, seed):
    """A deterministic pseudo-random value in [0, 1) per integer lattice point, from integer arithmetic alone: the same (ix, iy, iz,
    seed) hashes to the same value on any machine, whole frame or one tile at a time."""
    h = (ix.astype(np.int64) * np.int64(374761393) + iy.astype(np.int64) * np.int64(668265263)
         + iz.astype(np.int64) * np.int64(2147483647) + np.int64(int(seed)) * np.int64(2654435761))
    h = h & np.int64(0xffffffff)
    h = ((h ^ (h >> 13)) * np.int64(1274126177)) & np.int64(0xffffffff)
    h = h ^ (h >> 16)
    return h.astype(np.float64) / 4294967295.0


def _fade(t):
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def value_noise3(gx, gy, gz, seed):
    """3D value noise in [0, 1]: the lattice hash blended trilinearly with Perlin's quintic curve, so the field is smooth in z too."""
    ix, iy, iz = np.floor(gx), np.floor(gy), np.floor(gz)
    fx, fy, fz = _fade(gx - ix), _fade(gy - iy), _fade(gz - iz)
    ix, iy, iz = ix.astype(np.int64), iy.astype(np.int64), iz.astype(np.int64)

    def corner(dx, dy, dz):
        return _hash3(ix + dx, iy + dy, iz + dz, seed)

    def lerp(a, b, t):
        return a + t * (b - a)
    x00, x10 = lerp(corner(0, 0, 0), corner(1, 0, 0), fx), lerp(corner(0, 1, 0), corner(1, 1, 0), fx)
    x01, x11 = lerp(corner(0, 0, 1), corner(1, 0, 1), fx), lerp(corner(0, 1, 1), corner(1, 1, 1), fx)
    return lerp(lerp(x00, x10, fy), lerp(x01, x11, fy), fz)


def noise_field(p, x0, y0, w, h):
    """The Noise node's 0..1 field over absolute pixels x0.., y0.. (rows counted from the bottom).

    The pixel's position is taken back through the transform tab (translate, then rotate about the centre, skew, scale; counter-clockwise
    for a positive rotate with y up), divided by the x and y size into noise space, and turned by `xrotate` and `yrotate` about the x and y
    axes of the 3D volume that `zoffset` is the third coordinate of. `type` fBm sums `octaves` noise functions, each `lacunarity` times
    the frequency and `gain` times the amplitude of the last; turbulence sums their absolute value about the middle. `nyquist` stops
    adding octaves whose wavelength is shorter than two pixels. The sum is divided by the amplitudes' sum and raised to 1 / `gamma`."""
    from .core import split_values
    xs = np.arange(x0, x0 + w, dtype=np.float64)
    ys = rows(p, y0, h)
    xx, yy = np.meshgrid(xs, ys)
    cx, cy = float(p.get("center_x", 0.0)), float(p.get("center_y", 0.0))
    qx, qy = xx - cx - float(p.get("translate_x", 0.0)), yy - cy - float(p.get("translate_y", 0.0))
    theta = np.radians(float(p.get("rotate_x", 0.0)) + float(p.get("rotate_y", 0.0)))
    ca, sa = np.cos(theta), np.sin(theta)
    rx, ry = ca * qx + sa * qy, -sa * qx + ca * qy
    rx = rx - np.tan(np.radians(float(p.get("skew", 0.0)))) * ry
    scale = max(float(p.get("scaling", 1.0)), 1e-6)
    px, py = rx / scale + cx, ry / scale + cy
    size_x, size_y = (max(float(v), 1e-6) for v in split_values(p, "size", 2))
    gx, gy = px / size_x, py / size_y
    gz = np.full_like(gx, float(p.get("z_slice", 0.0)))
    a, b = np.radians(float(p.get("xrotate", 0.0))), np.radians(float(p.get("yrotate", 0.0)))
    x1, z1 = np.cos(b) * gx + np.sin(b) * gz, -np.sin(b) * gx + np.cos(b) * gz
    y2, z2 = np.cos(a) * gy - np.sin(a) * z1, np.sin(a) * gy + np.cos(a) * z1
    gx, gy, gz = x1, y2, z2
    seed = int(p.get("seed", 0))
    lacunarity, gain = float(p.get("lacunarity", 2.0)), float(p.get("gain", 0.5))
    gamma = max(float(p.get("gamma", 1.0)), 1e-6)
    turbulence = p.get("type", "fBm") == "turbulence"
    total = np.zeros_like(gx)
    amplitude, frequency, amplitudes = 1.0, 1.0, 0.0
    finest = max(1.0 / size_x, 1.0 / size_y)
    for octave in range(max(1, int(p.get("octaves", 1)))):
        if octave and int(p.get("nyquist", 1)) and finest * frequency > 0.5:
            break
        sample = value_noise3(gx * frequency, gy * frequency, gz * frequency, seed + octave * 101)
        if turbulence:
            sample = np.abs(2.0 * sample - 1.0)
        total = total + amplitude * sample
        amplitudes += amplitude
        amplitude *= gain
        frequency *= lacunarity
    return np.clip(total / max(amplitudes, 1e-9), 0.0, 1.0) ** (1.0 / gamma)


# ------------------------------------------------------------------------------------------------------- Dither and Grain
def dither_is_legacy(p):
    return bool(int(p.get("legacy_dither", 1)))


def grain_is_legacy(p):
    return bool(int(p.get("legacy_grain", 1)))


def dither(image, p, channels, origin, frame):
    """Nuke's Dither: each pixel gets a value between -0.5 and +0.5 times `amount` added, from a hash of its position, the channel (one
    pattern for all channels with monodither), the seed and the frame (the frame is left out with static seed). `bits` above zero rounds the
    result afterwards."""
    amount = float(p.get("dither_amount", 0.1))
    seed = int(p.get("seed", 0)) + (0 if int(p.get("static_seed", 0)) else int(frame or 0))
    mono = bool(int(p.get("monodither", 0)))
    bits = int(p.get("bits", 0))
    height, width = image.shape[:2]
    iy, ix = np.mgrid[0:height, 0:width]
    ix, iy = ix + int(origin[0]), iy + int(origin[1])
    out = image.copy()
    for c in channels:
        noise = _hash3(ix, iy, np.full_like(ix, 0 if mono else c), seed) - 0.5
        value = image[..., c].astype(np.float64) + noise * amount
        if bits > 0:
            levels = float(2 ** min(16, bits) - 1)
            value = np.clip(np.round(value * levels), 0.0, levels) / levels
        out[..., c] = value.astype(np.float32)
    return out


def grain(image, p, origin, frame, value_noise):
    """Nuke's Grain as far as its page states it: per-channel size, irregularity and intensity of a zero-mean noise added to the picture,
    scaled by the picture's brightness (`black` at black, 1 at white) and, with `Apply only through alpha`, by its alpha (the inverse with
    `Invert alpha`); the result is never lower than `minimum` rgb. A different pattern each frame, the same one when the seed is minus the frame."""
    height, width = image.shape[:2]
    xs = np.arange(width, dtype=np.float64) + int(origin[0])
    ys = np.arange(height, dtype=np.float64) + int(origin[1])
    gx, gy = np.meshgrid(xs, ys)
    seed = int(p.get("seed", 0)) + int(frame or 0)
    rgb = image[..., :3].astype(np.float64)
    luma = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]
    black = float(p.get("black", 0.0))
    weight = black + (1.0 - black) * np.clip(luma, 0.0, 1.0)
    if int(p.get("maskgrain", 1)):
        alpha = np.clip(image[..., 3].astype(np.float64), 0.0, 1.0)
        weight = weight * ((1.0 - alpha) if int(p.get("invert_mask", 0)) else alpha)
    out = image.copy()
    for c, name in enumerate(("red", "green", "blue")):
        intensity = float(p.get(f"{name}_intensity", 0.0)) * GRAIN_SCALE
        if intensity == 0.0:
            continue
        size = max(float(p.get(f"{name}_size", 1.0)), 1.0)
        bend = min(max(float(p.get(f"{name}_i", 0.0)), 0.0), 1.0)
        px, py = gx, gy
        if bend > 0.0:
            # The lattice is bent by a coarser noise of its own, as far as `bend` times the grain's size.
            px = gx + (value_noise(gx / (4.0 * size), gy / (4.0 * size), c + 3, seed + 1) - 0.5) * 2.0 * bend * size
            py = gy + (value_noise(gx / (4.0 * size), gy / (4.0 * size), c + 6, seed + 2) - 0.5) * 2.0 * bend * size
        noise = (value_noise(px / size, py / size, c, seed) - 0.5) * np.sqrt(12.0)
        out[..., c] = (image[..., c] + noise * intensity * weight).astype(np.float32)
    minimum = np.array([p.get("minimum_red", 0.0), p.get("minimum_green", 0.0), p.get("minimum_blue", 0.0)], dtype=np.float32)
    out[..., :3] = np.maximum(out[..., :3], minimum)
    return out


# ---------------------------------------------------------------------------------------------------- Defocus, ZDefocus, EdgeExtend
def defocus_radii(p):
    """Semi-axes (rx, ry) of Defocus's disc in pixels. The page: `defocus` is the disc's size, `aspect ratio` widens it without altering
    its height and `scaling` scales width and height (one number or a width and a height). An older node narrowed the height instead."""
    from . import core
    radius = abs(float(p.get("defocus", 6.0)))
    aspect = float(p.get("aspect", 1.0))
    if int(p.get("legacy_defocus", 1)):
        return radius, radius / aspect if aspect > 0 else radius
    sw, sh = (abs(float(v)) for v in core.split_values(p, "scale", 2, default=1.0))
    return radius * (aspect if aspect > 0 else 1.0) * sw, radius * sh


def zdefocus_distance(z, mode):
    """The distance from the camera that a depth value means under the page's `math` choice (direct and -direct have none)."""
    z = np.asarray(z, np.float32)
    if mode == "-depth":
        return -z
    if mode == "far = 0":
        return 1.0 / np.maximum(z, 1e-6)
    if mode == "far = -0":
        return 1.0 / np.maximum(-z, 1e-6)
    return z


def edge_extend_known(known, erode):
    """The trusted region `known` (bool) moved by `erode` pixels with a box: positive erodes it, negative dilates it."""
    n = int(round(abs(float(erode))))
    if n == 0:
        return known
    height, width = known.shape
    padded = np.pad(known, n, constant_values=bool(erode > 0))
    stack = [padded[n + dy:n + dy + height, n + dx:n + dx + width] for dy in range(-n, n + 1) for dx in range(-n, n + 1)]
    return np.logical_and.reduce(stack) if erode > 0 else np.logical_or.reduce(stack)
