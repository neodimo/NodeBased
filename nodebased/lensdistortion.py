"""LensDistortion as the Reference Guide describes it (lane 9, 2D behaviour step J): Undistort and Redistort read Nuke's Model Preset,
Lens Type, squeeze, numerator and denominator coefficients, centre, Mode, direction, BBox, Output Format and Distortion Scaling.

What the guide states, and what is read from it
* NukeX Classic is a Forward model, "xu = xd / (1 + k0 * rd^2 + k1 * rd^4)", with the k-values the two Denominator s knobs
  (`distortionDenominator0`, `distortionDenominator1`) and no numerator. A Forward model "needs to be inverted for image rendering";
  a Backward model "can be used directly". Custom shows the numerator too: Order and Exponent describe "the radial distortion's
  rational polynomial function", the first term for the numerator, the second for the denominator, so Custom reads
  (1 + n0 r^2 + n1 r^4) / (1 + d0 r^2 + d1 r^4) (exponent 2, the default; the guide gives the numerator's form only through the
  Order and Exponent text). `nodebased.lensmodel.RationalLens` is that function and its Newton inverse.
* Units. The guide's Normalisation (default Maximum) normalises "the focal length and distortion parameters"; it gives no
  factor, so r is measured in units of half the longer side of the format (r = 1 at the middle of the longer edge). Width,
  Height and Diagonal are not offered.
* Centre: "the position of the distortion center", default 0,0. Read here as pixels from the middle of the format, x to the right and y up.
* Lens Type Anamorphic: "Squeeze ... rescales the x coefficients", so x is divided by the squeeze before the model runs and
  multiplied after it, the same as the Brown-Conrady lens already did. A Spherical lens ignores the squeeze (the guide shows the
  anamorphic controls only for an Anamorphic lens). Twist and Scale (lens breathing) are not built.
* Mode: Undistort and Redistort warp the image (the Undistort kind is Nuke's LensDistortion; the Redistort kind is the same node
  with the mode fixed); STMap "renders both undistortion and redistortion STMaps in the motion channels ... the forward channels
  for undistorting and the backward channels for redistorting", the image itself copied through. NodeBased's motion layers are
  `vector.forward` and `vector.backward`.
* BBox: Auto grows to hold the warp (Undistort) or is the output format (Redistort); Input is the input's data window; Manual is
  x, y, r, t of the output format, y counted upwards.
* Output Format: Input is the input format; Format a named format; BBox the bounding box itself. Output formats are centred on the
  input format's middle.
* Distortion Scaling: Scale to Input Format measures r against the input format, Format against the chosen one.

Model presets other than the three offered ("CaraVR Radial, Degree 3" and "3DEqualizer's" models) are named by the guide without
their equations and are not offered. The Brown-Conrady preset is NodeBased's own older lens (the `lens_*` knobs and `lens_preset`):
documents saved before this step carry it, so they render as before. A wired camera always supplies its own Brown-Conrady lens.

Pure NumPy, so the evaluator and the tests read one arithmetic.
"""
from __future__ import annotations

import numpy as np

CLASSIC = "NukeX Classic"
CUSTOM = "Custom"
BROWN = "Brown-Conrady (NodeBased)"
PRESETS = (CLASSIC, CUSTOM, BROWN)
LENS_TYPES = ("Spherical", "Anamorphic")
OUTPUTS = ("Undistort", "Redistort", "STMap")
DIRECTIONS = ("Forward", "Backward")
BBOX_TYPES = ("Auto", "Input", "Manual")
FORMAT_TYPES = ("Input", "Format", "BBox")
SCALING_TYPES = ("Scale to Input Format", "Format")
MODEL_TYPES = ("Radial Standard", "Radial Asymmetric", "Radial-Tangential (Coupled)", "Radial-Tangential (Uncoupled)")
NORMALISATIONS = ("Maximum", "Width", "Height", "Diagonal")
PROJECTIONS = ("None (Rectilinear)", "Stereographic", "Equidistant", "Equisolid", "Orthographic")

KINDS = ("Undistort", "Redistort")
_COMMON = {"distortionModelPreset": CLASSIC, "lens_type": "Spherical", "distortionNumerator0": 0.0, "distortionNumerator1": 0.0,
           "distortionDenominator0": 0.0, "distortionDenominator1": 0.0, "centre_x": 0.0, "centre_y": 0.0,
           "distortionModelDirection": "Forward", "bboxType": "Auto", "bbox_x": 0.0, "bbox_y": 0.0, "bbox_r": 1920.0, "bbox_t": 1080.0,
           "outputFormatType": "Input", "lens_format": "HD_1080", "distortionScalingType": "Scale to Input Format",
           "lens_scaling_format": "HD_1080", "anamorphicTwist": 0.0, "anamorphicScale_x": 1.0, "anamorphicScale_y": 1.0,
           "distortionModelType": "Radial Standard", "distortionOrder_x": 0, "distortionOrder_y": 2,
           "distortionExponent_x": 2, "distortionExponent_y": 2, "normalisationType": "Maximum",
           "distortInFisheyeSpace": 1, "projection": "None (Rectilinear)", "focal": 9.0, "sensorSize_x": 36.0,
           "sensorSize_y": 24.0, "useFisheye": 1, "enableBeamSplitter": 0, "beamSplitterDirection": 0.0,
           "beamSplitterBending": 0.0, "distortionModelDisplayX": "xu = xd / (1 + k0 * rd^2 + k1 * rd^4)",
           "distortionModelDisplayY": "yu = yd / (1 + k0 * rd^2 + k1 * rd^4)"}
NEW = {"Undistort": {**_COMMON, "lens_output": "Undistort"}, "Redistort": dict(_COMMON)}


def new_params(kind):
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What a node saved before this step gets: the Brown-Conrady preset (its own lens knobs), every other knob at the value that
    leaves the picture as it was."""
    if kind not in NEW:
        return {}
    return {**NEW[kind], "distortionModelPreset": BROWN}


def _rect_size(name):
    from . import core
    w, h, _aspect = core.REFORMAT_FORMATS.get(name, core.REFORMAT_FORMATS["HD_1080"])
    return int(w), int(h)


def preset_of(p):
    """The model preset of a parameter dict; one that predates the knob is an older document's node: the Brown-Conrady lens."""
    return p.get("distortionModelPreset", BROWN)


def mode_of(kind, p):
    return "Redistort" if kind == "Redistort" else p.get("lens_output", "Undistort")


def build_lens(p, display, lensmodel):
    """(lens, frame) for the node's own knobs (no camera): `frame` is None, or the (width, height, left, top) of the Distortion
    Scaling format centred on the input format."""
    scaling = p.get("distortionScalingType", SCALING_TYPES[0])
    frame = None
    fw, fh = display.width, display.height
    if scaling == "Format":
        fw, fh = _rect_size(p.get("lens_scaling_format", "HD_1080"))
        frame = (fw, fh, display.x + (display.width - fw) / 2.0, display.y + (display.height - fh) / 2.0)
    preset = preset_of(p)
    if preset == BROWN:
        if p.get("lens_preset", "custom") == "custom":
            lens = lensmodel.from_params(p, p.get("anamorphic_squeeze", 1.0))
        else:
            lens = lensmodel.Lens(**{name[5:]: value for name, value in lensmodel.PRESETS[p["lens_preset"]].items()},
                                  squeeze=float(p.get("anamorphic_squeeze", 1.0)) or 1.0)
        return lens, frame
    half = max(float(fh), 1.0) / 2.0
    anamorphic = p.get("lens_type", "Spherical") == "Anamorphic"
    squeeze = (float(p.get("anamorphic_squeeze", 1.0)) or 1.0) if anamorphic else 1.0
    numerator = (float(p.get("distortionNumerator0", 0.0)), float(p.get("distortionNumerator1", 0.0))) if preset == CUSTOM else (0.0, 0.0)
    lens = lensmodel.RationalLens(numerator[0], numerator[1], float(p.get("distortionDenominator0", 0.0)),
                                  float(p.get("distortionDenominator1", 0.0)),
                                  p.get("distortionModelDirection", "Forward") == "Forward" if preset == CUSTOM else True,
                                  float(p.get("centre_x", 0.0)) / half, float(p.get("centre_y", 0.0)) / half, squeeze,
                                  float(fh) / max(float(fw), float(fh), 1.0),
                                  float(p.get("anamorphicTwist", 0.0)) if anamorphic else 0.0,
                                  float(p.get("anamorphicScale_x", 1.0)) if anamorphic else 1.0,
                                  float(p.get("anamorphicScale_y", 1.0)) if anamorphic else 1.0)
    return lens, frame


def output_display(p, display, data):
    """The output's format window: the input's, a named format or (BBox) the bounding box itself, the named one centred on the input's."""
    kind = p.get("outputFormatType", "Input")
    if kind == "Format":
        w, h = _rect_size(p.get("lens_format", "HD_1080"))
        from .tiers import Region
        return Region(int(display.x + (display.width - w) // 2), int(display.y + (display.height - h) // 2), w, h)
    return display


def manual_box(p, display):
    """The Manual BBox as an image-space rectangle (x, y, w, h): x, y, r, t of the format, y counted upwards."""
    from .tiers import Region
    x, y, r, t = (float(p.get(n, 0.0)) for n in ("bbox_x", "bbox_y", "bbox_r", "bbox_t"))
    return Region(int(round(display.x + x)), int(round(display.y + display.height - t)), max(0, int(round(r - x))), max(0, int(round(t - y))))


def stmap_layers(lens, out, display, source_data, frame, lensmodel, Raster):
    """The forward (undistorting) and backward (redistorting) STMaps over `out`, as NodeBased's `vector.forward` and `vector.backward`.

    An STMap holds, for each output pixel centre, the position to read in the input as (u, v): u across the input format from 0 to 1
    and v up from 0 to 1 (the convention of the STMap node), so the identity map holds each pixel's own centre."""
    width, height = display.width, display.height
    layers = {}
    for name, direction in (("vector.forward", "undistort"), ("vector.backward", "redistort")):
        sx, sy = lensmodel.sample_coordinates(lens, out, display, source_data, direction, frame)
        u = (sx.astype(np.float64) + 0.5 + source_data.x - display.x) / width
        v = 1.0 - (sy.astype(np.float64) + 0.5 + source_data.y - display.y) / height
        rgba = np.zeros((out.height, out.width, 4), np.float32)
        rgba[..., 0], rgba[..., 1], rgba[..., 3] = u, v, 1.0
        layers[name] = Raster(rgba, out, display)
    return layers
