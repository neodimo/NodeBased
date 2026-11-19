"""The Bokeh node (lane 9, 2D behaviour step X1), as the Nuke 17.0 Reference Guide states it.

Bokeh "defocuses an image according to a Z depth map, Deep data, or Camera information. You can control where the focal plane lies to focus on
specific elements in an image and simulate real-world lenses. You can also control the shape of the defocus kernel using the Kernel input."
The page names each knob's effect and gives no formula, so the numbers below are ours; every sentence quoted is the guide's. The blur is
gathered by ZDefocus's own kernels (`Evaluator._zdefocus`: the disc, the bladed polygon, the kernel image, their occlusion rule), which are handed
Bokeh's per-pixel blur sizes; Bokeh adds the depth reading, the focus region, the bloom and the outputs.

* Multiplier (`multiplier`, `frontmultiplier`, `backmultiplier`): "the overall amount of defocus", and "the amount of defocus applied to the front
  [back] focus region in addition to the defocus applied by the Multiplier control". Ours: the blur radius is BLUR_PER_MULTIPLIER (10) pixels times
  the multiplier times the region's multiplier times the blur amount below. "You can defocus an image without using any additional depth data,
  but the result is uniformly defocused by the Multiplier control": with no depth channel the radius is 10 * multiplier everywhere.
* Focal plane, focus region size and falloff (`focalPlane`, `focusRegionSize`, `focusRegionFalloff`): "the plane on which the image is totally in
  focus", "the size of the plane totally in focus. Larger values increase the width of the plane": the region is focalPlane +- size / 2. Outside it
  the blur amount is the distance past the region's edge over the pixel's own distance (the thin-lens relation: it tends to 1 far behind and
  exceeds 1 in front), shaped by the falloff ("applied between the focal plane and the front and back regions"): Linear is the amount itself
  (the guide's default), Smooth is a smooth step of it (ours; the page names no other choice).
* Depth style (`depthStyle`): "Real - uses real world values", "1/z - uses values similar to those output by ScanlineRender": the distance is the
  depth, or one over it.
* Real world lens (`realWorldLens`, `focalLength`, `fStop`, `worldScale`, `worldScaleMultiplier`, `filmFormat`, `apertureWidth`, `apertureHeight`):
  "when enabled, you can use the Lens controls to match your physical lens characteristics ... when disabled, the Lens controls have no effect". On,
  the blur is the circle of confusion of a thin lens, c = f^2 |d - s| / (N d (s - f)) (the standard optics formula, a "convention" source),
  with the distances in millimetres (world scale times its multiplier; mm, cm, m, in and ft are ours, the page names none), s the near or far edge
  of the focus region, and its radius c / 2 turned into pixels by the film width: the frame's width over the aperture width in millimetres. The
  35mm film format is the guide's 22 x 16 aperture; Custom lets the aperture be typed ("set Film Format to Custom and then enter the required
  aperture dimensions in mm"); the page lists no other format.
* Outputs (`outputType`): Defocused Image; Lens Bloom ("just the bloom effect"); Focal Distance Visualization ("Red - the focal plane, where your
  image is fully in focus. Green - areas of the image in front of the focal plane. Blue - areas of the image behind the focal plane"; ours: red is
  one minus the blur amount, green the amount in front, blue the amount behind, limited to 0 to 1 when `normalizeVisualization` is on, "color
  values in the output are limited to the 0-1 range"); Filter Shape ("preview the shape of the kernel", drawn white and centred at the largest
  size); Focus Mattes ("Front/Back mattes for the focus": ours, front in red, back in green, each the amount limited to 0 to 1). The page lists
  Red, Green and Blue as outputs without saying what they show and they are not offered.
* Focus region output (`focusOutput`): Both, Front or Back: "output to the Viewer ... without taking the processing hit for both regions". Ours: a
  pixel outside the chosen region keeps its input colour.
* Matte channels (`defocusMatteChannel`, `bloomMatteChannel`, `chromicAberrationMatteChannel`): "the channel to use as a mask. By default, the
  defocus effect is limited to the non-black areas of this channel" (and likewise the bloom and chromatic aberration): the effect is mixed in by
  the channel's value; "none" is no mask.
* Bloom (`bokeh_bloom`, `bloomThreshold`, `bloomCurvature`): "the brightness of any bloom applied to the image when the Bloom Threshold is reached. Any
  pixels with a brightness less than the Bloom Threshold are not affected". Ours: the bloom is bloom times the part of each colour above the
  threshold, gathered through the same filter at the pixel's blur size and added to the result.
* Spherical aberration (`kSphAbb`, `bloomCurvature`): "Negative Spherical Aberration values make the kernels brighter in the center and positive
  values brighter on the edges." Ours: the filter's weight at a fraction q of its radius is 1 + k (2 q^curvature - 1), cut at 0; the page says only
  that the bloom curvature "defines the physical lens curve, which affects the bloom differently depending where light enters the lens", so the
  curvature is the exponent of that profile and acts only with a spherical aberration.
* Chromatic aberration (`kChrAbb`, `kChrAbbOff`): the page says it "affects the bloom color" and that the offset "controls the chromatic refraction
  offset for the red, green, and blue channels". Ours: the bloom of each colour is gathered at a radius scaled by 1 + k (offset - 1); with the
  default offsets 0.6, 1, 1 red is the smaller. Not scored: the page gives no numbers.
* Kernel (`kernelType`, `max_kernelsize`, `kAspectRatio`, `kSoftness`, `kNumSides`, `kCurvature`, `kRotation`, `missingKernelChannel`): Input is the
  wired kernel image (ZDefocus's image filter), Circular a round filter, Aperture Blades the polygon of Blade Count sides ("any value between 3 and
  16"). Softness "0 produces a very hard edge and a value of 1 produces a very soft edge": the weight of the disc is dissolved into a Gaussian by that
  amount (ZDefocus's `shape` is one minus it). Curvature "lower values produce sharper angles on the kernel shape and higher values smooth the kernel
  shape" and Rotation turn the polygon (ZDefocus's roundness and rotation). Aspect Ratio is the filter's width over its height. Maximum Kernel Size
  "sets an upper limit on the size of the kernel": a diameter, so a radius of half of it. Missing Kernel Channel: the channel read when the kernel
  input's alpha is empty; it "has no effect on the Circular and Aperture Blades presets".
* Corrective slices (`correctiveSlices`, `overrideNearAndFarForSlices`, `slices_near`, `slices_far`): "the number of slices the depth channel is
  divided into during defocus calculations, which can help avoid objects at different depths overlapping incorrectly. The depth volume sliced is
  defined by the Focal Plane and Focus Region Size": 1 slice, no ordering (pixels gather from any depth: the overlap); 2, the front and back processed
  separately; more, that many equal slices of the volume (a pixel gathers from its own slice and nearer ones). Override Near/Far replaces the volume.
* Not provided: the Deep and Camera inputs, Integrate Deep Front and Back Separately and Expand Bounding Box To Max Kernel Size (NodeBased images
  keep their format and there is no Deep image); the mapping lists each as unsupported.
"""
from __future__ import annotations

import math

import numpy as np

BLUR_PER_MULTIPLIER = 10.0
OUTPUT_TYPES = ("Defocused Image", "Lens Bloom", "Focal Distance Visualization", "Filter Shape", "Focus Mattes")
FOCUS_OUTPUTS = ("Both", "Front", "Back")
DEPTH_STYLES = ("Real", "1/z")
FALLOFFS = ("Linear", "Smooth")
KERNEL_TYPES = ("Input", "Circular", "Aperture Blades")
WORLD_SCALES = {"mm": 1.0, "cm": 10.0, "m": 1000.0, "in": 25.4, "ft": 304.8}
FILM_FORMATS = ("35mm", "Custom")
FILM_APERTURES = {"35mm": (22.0, 16.0)}
NEW = {"depthChannel": "depth.Z", "outputType": "Defocused Image", "focusOutput": "Both", "multiplier": 1.0, "frontmultiplier": 1.0,
       "backmultiplier": 1.0, "depthStyle": "Real", "focalPlane": 100.0, "focusRegionSize": 1.0, "focusRegionFalloff": "Linear",
       "defocusMatteChannel": "none", "normalizeVisualization": 1,
       "realWorldLens": 0, "focalLength": 35.0, "fStop": 6.0, "worldScale": "cm", "worldScaleMultiplier": 1.0, "filmFormat": "35mm",
       "apertureWidth": 22.0, "apertureHeight": 16.0,
       "bokeh_bloom": 0.0, "bloomCurvature": 2.0, "bloomThreshold": 0.7, "bloomMatteChannel": "none", "kSphAbb": 0.0, "kChrAbb": 0.0,
       "chromic_offset_red": 0.6, "chromic_offset_green": 1.0, "chromic_offset_blue": 1.0, "chromicAberrationMatteChannel": "none",
       "kernelType": "Circular", "max_kernelsize": 256.0, "missingKernelChannel": "none", "kAspectRatio": 1.0, "kSoftness": 0.1, "kNumSides": 6,
       "kCurvature": 0.1, "kRotation": 0.0,
       "correctiveSlices": 10, "overrideNearAndFarForSlices": 0, "slices_near": 0.0, "slices_far": 0.0}
LIMITS = {"multiplier": (0.0, 1000.0), "frontmultiplier": (0.0, 1000.0), "backmultiplier": (0.0, 1000.0), "focalPlane": (-1.0e6, 1.0e6),
          "focusRegionSize": (0.0, 1.0e6), "normalizeVisualization": (0, 1), "realWorldLens": (0, 1), "focalLength": (0.1, 10000.0),
          "fStop": (0.1, 128.0), "worldScaleMultiplier": (1.0e-6, 1.0e6), "apertureWidth": (0.01, 1000.0), "apertureHeight": (0.01, 1000.0),
          "bokeh_bloom": (0.0, 1000.0), "bloomCurvature": (0.0, 20.0), "bloomThreshold": (-1000.0, 1000.0), "kSphAbb": (-1.0, 1.0),
          "kChrAbb": (-10.0, 10.0), "chromic_offset_red": (-10.0, 10.0), "chromic_offset_green": (-10.0, 10.0),
          "chromic_offset_blue": (-10.0, 10.0), "max_kernelsize": (0.0, 4096.0), "kAspectRatio": (0.1, 10.0), "kSoftness": (0.0, 1.0),
          "kNumSides": (3, 16), "kCurvature": (0.0, 1.0), "kRotation": (-3600.0, 3600.0), "correctiveSlices": (1, 256),
          "overrideNearAndFarForSlices": (0, 1), "slices_near": (-1.0e6, 1.0e6), "slices_far": (-1.0e6, 1.0e6)}
CHOICES = {"outputType": list(OUTPUT_TYPES), "focusOutput": list(FOCUS_OUTPUTS), "depthStyle": list(DEPTH_STYLES),
           "focusRegionFalloff": list(FALLOFFS), "worldScale": list(WORLD_SCALES), "filmFormat": list(FILM_FORMATS),
           "kernelType": list(KERNEL_TYPES)}
MATTE_PARAMS = ("defocusMatteChannel", "bloomMatteChannel", "chromicAberrationMatteChannel")
_WORDS = {"red": 0, "r": 0, "green": 1, "g": 1, "blue": 2, "b": 2, "alpha": 3, "a": 3}


def needs_full_frame(p):
    """The kernel image and the filter preview are whole pictures, and the lens reads the frame's width: tiles cannot do any of the three."""
    return p.get("kernelType") == "Input" or p.get("outputType") == "Filter Shape" or bool(int(p.get("realWorldLens", 0)))


def reach(p):
    """The widest the filter can be, in pixels from its middle: half the maximum kernel size, rounded up."""
    return int(math.ceil(max(0.0, abs(float(p.get("max_kernelsize", 256.0))) / 2.0)))


def matte_plane(pixels, name, named):
    """The matte channel `name` of a picture (None when it is "none"): an rgba word, an `rgba.alpha` style name, or a layer through `named`."""
    if not name or str(name).lower() == "none":
        return None
    low = str(name).lower()
    if low in _WORDS:
        return pixels[..., _WORDS[low]]
    if "." in low and low.split(".", 1)[0] in ("rgba", "rgb") and low.rsplit(".", 1)[1] in _WORDS:
        return pixels[..., _WORDS[low.rsplit(".", 1)[1]]]
    got = named(name)
    if got is None:
        raise ValueError(f"Bokeh: missing channel {name!r}")
    got = np.asarray(got)
    return got[..., 0] if got.ndim == 3 else got


def distance(depth, p):
    """The depth as a distance: real world values as they are, 1/z as one over them."""
    d = np.asarray(depth, np.float64)
    if p.get("depthStyle", "Real") == "1/z":
        d = 1.0 / np.maximum(d, 1e-6)
    return d


def _falloff(t, p):
    t = np.asarray(t, np.float64)
    if p.get("focusRegionFalloff", "Linear") == "Smooth":
        s = np.clip(t, 0.0, 1.0)
        return s * s * (3.0 - 2.0 * s) + (t - s)
    return t


def amounts(dist, p):
    """(front amount, back amount, edge below, edge above): the blur amount of each pixel by the region it lies in, and the region's edges."""
    c, half = float(p.get("focalPlane", 100.0)), max(0.0, float(p.get("focusRegionSize", 1.0))) / 2.0
    below, above = c - half, c + half
    ref = np.maximum(np.abs(dist), 1e-6)
    front = _falloff(np.maximum(below - dist, 0.0) / ref, p)
    back = _falloff(np.maximum(dist - above, 0.0) / ref, p)
    return front, back, below, above


def lens_radius(dist, below, above, p, width):
    """The real world lens: the thin-lens circle of confusion at each pixel, as a radius in pixels (see the module's note)."""
    scale = WORLD_SCALES.get(p.get("worldScale", "cm"), 10.0) * max(1e-9, float(p.get("worldScaleMultiplier", 1.0)))
    f, n = float(p.get("focalLength", 35.0)), max(0.1, float(p.get("fStop", 6.0)))
    d = np.maximum(dist, 1e-6) * scale
    s = np.where(dist < below, below, above) * scale
    s = np.maximum(s, f * 1.0001)
    coc = f * f * np.abs(d - s) / (n * d * (s - f))
    if p.get("filmFormat", "35mm") == "Custom":
        film = max(1e-6, float(p.get("apertureWidth", 22.0)))
    else:
        film = FILM_APERTURES.get(p.get("filmFormat", "35mm"), (22.0, 16.0))[0]
    inside = (dist >= below) & (dist <= above)
    return np.where(inside, 0.0, coc / 2.0 * (width / film))


def blur(dist, p, width, height):
    """(radii in pixels, sides, front amount, back amount): the filter's radius at each pixel (capped at half the maximum kernel size),
    -1 where the pixel lies in front of the focus region, 1 behind it, 0 in it; with no depth the radius is uniform and the side 0."""
    cap = abs(float(p.get("max_kernelsize", 256.0))) / 2.0
    mult = float(p.get("multiplier", 1.0))
    if dist is None:
        radii = np.full((height, width), BLUR_PER_MULTIPLIER * mult, np.float64)
        zeros = np.zeros((height, width), np.float64)
        return np.minimum(radii, cap).astype(np.float32), np.zeros((height, width), np.int64), zeros, zeros
    front, back, below, above = amounts(dist, p)
    sides = np.where(dist < below, -1, np.where(dist > above, 1, 0))
    if int(p.get("realWorldLens", 0)):
        amount = lens_radius(dist, below, above, p, width)
        radii = amount * mult * np.where(sides < 0, float(p.get("frontmultiplier", 1.0)), float(p.get("backmultiplier", 1.0)))
    else:
        radii = BLUR_PER_MULTIPLIER * mult * (front * float(p.get("frontmultiplier", 1.0)) + back * float(p.get("backmultiplier", 1.0)))
    return np.minimum(radii, cap).astype(np.float32), sides, front, back


def occlusion(dist, p):
    """The ordering depth the corrective slices make: constant for one slice, else the slice number of the depth volume."""
    n = max(1, int(p.get("correctiveSlices", 10)))
    if n <= 1:
        return np.zeros(dist.shape, np.float32)
    if int(p.get("overrideNearAndFarForSlices", 0)):
        near, far = float(p.get("slices_near", 0.0)), float(p.get("slices_far", 0.0))
    else:
        c, half = float(p.get("focalPlane", 100.0)), max(0.0, float(p.get("focusRegionSize", 1.0))) / 2.0
        near, far = c - half, c + half
    near, far = min(near, far), max(near, far)
    if far - near < 1e-12:
        return (dist > near).astype(np.float32)
    return np.clip(np.floor((dist - near) / (far - near) * n), 0, n - 1).astype(np.float32)


def zdefocus_params(p, radius):
    """The ZDefocus parameters that make `_zdefocus` gather through Bokeh's kernel: the disc or the polygon, the filter image, softness and aspect."""
    kernel = p.get("kernelType", "Circular")
    return {"zdefocus_math": "depth", "legacy_zdefocus": 0, "legacy_zdefocus_focus": 0, "legacy_zdefocus_filters": 0, "focal_plane": 0.0,
            "depth_of_field": 0.0, "blur_dof": 0, "size": 1.0, "max_size": float(radius), "autoLayerSpacing": 1, "nLayers": 20, "layerCurve": 1.0,
            "bokeh_shape": {"Input": "image", "Circular": "disc", "Aperture Blades": "blades"}.get(kernel, "disc"),
            "shape": 1.0 - min(1.0, max(0.0, float(p.get("kSoftness", 0.1)))), "aspect": float(p.get("kAspectRatio", 1.0)), "catadioptric": 0,
            "catadioptric_size": 0.3, "bloom": 0, "bloom_threshold": 0.8, "bloom_gain": 2.0, "bloom_gamma": 0,
            "blade_count": max(3, min(16, int(p.get("kNumSides", 6)))), "blade_rotation": float(p.get("kRotation", 0.0)),
            "roundness": float(p.get("kCurvature", 0.1)), "inner_size": 0.5, "inner_feather": 0.5, "inner_brightness": 1.0,
            "legacy_resize_mode": 0, "filter_channel": "rgba.alpha", "use_input_channels": 0, "filter_bounds": "shape", "image_filter": "cubic",
            "clamp_image_filter": 0, "fill_foreground": 1, "zdefocus_output": "result", "show_image": 1, "focal_x": 200.0, "focal_y": 200.0}


def _gather(image, z, radii, sides, p, kernel):
    from .imaging import Evaluator
    zp = zdefocus_params(p, max(1.0, abs(float(p.get("max_kernelsize", 256.0))) / 2.0))
    if p.get("kernelType") == "Input" and kernel is not None and p.get("missingKernelChannel", "none") not in ("none", ""):
        if not np.any(kernel[..., 3] > 0):
            zp["filter_channel"] = str(p["missingKernelChannel"])
    sph = (float(p["kSphAbb"]), max(1e-3, float(p.get("bloomCurvature", 2.0)))) if float(p.get("kSphAbb", 0.0)) else None
    return Evaluator._zdefocus(image, z, zp, kernel, override={"z": z, "radii": radii, "sides": sides, "sph": sph})


def render(image, depth, p, kernel=None, planes=None):
    """The Bokeh node on a picture (H x W x 4) and its depth plane (H x W, or None): the picture or the chosen output.

    `planes` maps each of the three matte parameters to its plane (or None), read by the caller from the layers it can see."""
    image = np.asarray(image, np.float32)
    planes = planes or {}
    h, w = image.shape[:2]
    dist = None if depth is None else distance(depth, p)
    output = p.get("outputType", "Defocused Image")
    radii, sides, front, back = blur(dist, p, w, h)
    if output == "Focal Distance Visualization":
        amount = np.zeros((h, w)) if dist is None else np.maximum(front, back)
        out = np.ones(image.shape, np.float32)
        out[..., 0] = 1.0 - np.minimum(amount, 1.0)
        out[..., 1] = front if dist is not None else 0.0
        out[..., 2] = back if dist is not None else 0.0
        if int(p.get("normalizeVisualization", 1)):
            out[..., :3] = np.clip(out[..., :3], 0.0, 1.0)
        return out
    if output == "Focus Mattes":
        out = np.ones(image.shape, np.float32)
        out[..., 0] = np.clip(front, 0.0, 1.0)
        out[..., 1] = np.clip(back, 0.0, 1.0)
        out[..., 2] = 0.0
        return out
    z = np.zeros((h, w), np.float32) if dist is None else occlusion(dist, p)
    if output == "Filter Shape":
        from .imaging import Evaluator
        zp = zdefocus_params(p, max(1.0, abs(float(p.get("max_kernelsize", 256.0))) / 2.0))
        zp["zdefocus_output"] = "filter shape setup"
        return Evaluator._zdefocus(image, z, zp, kernel, override={"z": z, "radii": radii, "sides": sides, "sph": None})
    result = _gather(image, z, radii, sides, p, kernel)
    focus = p.get("focusOutput", "Both")
    if focus != "Both":
        keep = (sides >= 0) if focus == "Front" else (sides <= 0)
        result = np.where(keep[..., None], image, result)
    matte = planes.get("defocusMatteChannel")
    if matte is not None:
        m = np.clip(np.asarray(matte, np.float32), 0.0, 1.0)[..., None]
        result = image + (result - image) * m
    layer = None
    if float(p.get("bokeh_bloom", 0.0)) != 0.0 or output == "Lens Bloom":
        layer = bloom_layer(image, z, radii, sides, p, kernel, planes)
    if output == "Lens Bloom":
        out = image.copy()
        out[..., :3] = layer
        return out
    if layer is not None:
        result = result.copy()
        result[..., :3] += layer
    return result.astype(np.float32)


def bloom_layer(image, z, radii, sides, p, kernel, planes):
    """The bloom: bloom times the colour above the threshold, gathered through the filter at each pixel's size, a radius per colour with chromatic
    aberration; limited by the bloom matte, with the chromatic aberration limited by its own."""
    amount = float(p.get("bokeh_bloom", 0.0))
    excess = image.copy()
    excess[..., :3] = np.maximum(image[..., :3] - float(p.get("bloomThreshold", 0.7)), 0.0)
    excess[..., 3] = 0.0
    plain = _gather(excess, z, radii, sides, p, kernel)[..., :3]
    layer = plain
    k = float(p.get("kChrAbb", 0.0))
    if k != 0.0:
        cap = abs(float(p.get("max_kernelsize", 256.0))) / 2.0
        offsets = (float(p.get("chromic_offset_red", 0.6)), float(p.get("chromic_offset_green", 1.0)), float(p.get("chromic_offset_blue", 1.0)))
        coloured = np.empty_like(plain)
        for c, off in enumerate(offsets):
            scaled = np.minimum(radii * max(0.0, 1.0 + k * (off - 1.0)), cap).astype(np.float32)
            coloured[..., c] = _gather(excess, z, scaled, sides, p, kernel)[..., c]
        matte = planes.get("chromicAberrationMatteChannel")
        if matte is not None:
            m = np.clip(np.asarray(matte, np.float32), 0.0, 1.0)[..., None]
            coloured = plain + (coloured - plain) * m
        layer = coloured
    layer = layer * amount
    matte = planes.get("bloomMatteChannel")
    if matte is not None:
        layer = layer * np.clip(np.asarray(matte, np.float32), 0.0, 1.0)[..., None]
    return layer.astype(np.float32)


def planes_for(pixels, p, named):
    """The three matte planes of a node, read through `named` (layer name -> plane or None) by whichever path runs it."""
    return {name: matte_plane(pixels, p.get(name, "none"), named) for name in MATTE_PARAMS}
