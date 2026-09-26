"""Serializable graph and the shared human/agent command boundary (no Qt imports)."""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from . import expressions as expr
from . import filmback
from . import shapes

# Parameter schemas are also consumed by the inspector and agent discovery.
# Filter nodes accept an optional "mask" image (alpha gates where the filter applies) and a
# per-node "mix" (blend between original input and filtered output). The mask slot is listed in
# "optional_inputs" rather than "inputs" so a node validates without it wired — the evaluator
# treats None there as full opacity (M.a = 1).
# Position, BlackOutside and AdjustBBox (step 3b) change only *where* an image's data window sits,
# never its pixels' values, and Nuke gives them no mask or mix, so they are not MASK_MIX_KINDS: the
# evaluator handles them in one place (`Evaluator._window_node`), before the mask/mix branch.
WINDOW_KINDS = ("Position", "BlackOutside", "AdjustBBox")

IMAGE_FILTER_KINDS = ("Grade", "ColorCorrect", "Blur", "Transform", "Crop")

# Kinds that honour the optional-mask + mix contract. IMAGE_FILTER_KINDS is frozen history — the
# v3 -> v4 upgrade is written against it — so a kind that adopts the contract later joins this
# list instead, which is what the inspector and the evaluator read.
MASK_MIX_KINDS = IMAGE_FILTER_KINDS + ("Tracker", "Invert", "Clamp", "Multiply", "Add", "Gamma",
                                       "Saturation", "Erode", "Dilate", "Median", "Sharpen", "Glow", "Soften",
                                       "Defocus", "DirBlur", "DropShadow", "EdgeBlur", "EdgeExtend", "LightWrap", "Dither",
                                       "Grain", "Posterize", "SoftClip", "HSVTool", "Blend",
                                       "Exposure", "HueCorrect", "ColorMatrix",
                                       "Mirror", "Keyer", "HueKeyer", "Reformat", "CornerPin",
                                       "STMap", "IDistort", "VectorBlur")

# Kinds driven by a per-pixel two-channel map (step 5c). Their slots are image, uv, mask, in that
# order; they run on the whole-image path only (docs/PARITY_2D.md).
UV_KINDS = ("STMap", "IDistort", "VectorBlur")

# Reformat's node-local named-format presets (step 2c5's escape hatch from the document-level
# format registry the audit sketched -- see the SPECS["Reformat"] comment). Selecting one of these
# through a "set" command on the "format" param resolves it into width/height/pixel_aspect right
# there (`Dispatcher._edit`'s "set" branch), so the node's own params stay the single source of
# truth the kernel and the proxy-tier scaler read; "Custom" leaves width/height/pixel_aspect alone.
REFORMAT_FORMATS = {
    "HD_1080": (1920, 1080, 1.0),
    "HD_720": (1280, 720, 1.0),
    "UHD_4K": (3840, 2160, 1.0),
    "2K_DCP": (2048, 1080, 1.0),
    "Square_1K": (1024, 1024, 1.0),
}

# Two-input A/B kinds sharing Merge's bypass and windowing convention: bypass passes B (the
# background), or A when B is unwired; the union of A's and B's data windows is the output; an
# optional mask aligns to B's display window. See `bypass_slot` and `imaging._windowed_kernel`.
MERGE_LIKE_KINDS = ("Merge", "Dissolve", "Keymix", "Copy", "ChannelMerge", "Difference", "AddMix", "CopyRectangle")

# Draw-menu generators: own format (width/height), plus an optional "image" input the shape is
# composited over and an optional "mask". Bypassing one passes that optional image through (or a
# transparent format-sized frame when it is unwired) rather than the empty-slot behaviour a pure
# generator like Constant/Checker/Roto gets, because a Draw node's whole point is to sit inline in
# a chain over an existing plate. See `bypass_slot` and `imaging.Evaluator._windowed_kernel`.
DRAW_KINDS = ("Ramp", "Radial", "Rectangle", "Noise", "Text", "Grid")

# The version `upgrade_document` migrates to and `validate` accepts. Tests and callers should refer
# to this rather than hard-coding a number, so a schema bump does not spray stale literals.
SCHEMA_VERSION = 12
# Node-tab fields (Nuke's "Node" tab). Both are optional on a node and absent means default, so
# a comp has one serialized form: a node only carries them once an artist changed them.
NODE_LABEL_LIMIT = 1024
# Sources are where a postage stamp tells you something; on a filter it mostly repeats the input.
DEFAULT_THUMBNAIL_TYPES = ("Read", "ReadBundle", "Constant", "Checker")

# Connection typing is deliberately small and explicit.  A Render3D node is the only bridge
# from scene/camera values to the existing image graph; this prevents a malformed graph from
# failing much later inside a renderer.
# The full Nuke-style transform: T(translate) @ T(pivot) @ R(rot_order) @ S * uscale @ T(-pivot).
_XFORM = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0,
          "sx": 1.0, "sy": 1.0, "sz": 1.0, "uscale": 1.0, "rot_order": "XYZ",
          "pivot_x": 0.0, "pivot_y": 0.0, "pivot_z": 0.0}
# Knobs every particle force shares (step 2b). `seed` picks which particles `probability` selects.
_FORCE = {"probability": 1.0, "from_frame": -1000000, "to_frame": 1000000, "seed": 0}
_XFORM_ADDED = ("uscale", "rot_order", "pivot_x", "pivot_y", "pivot_z")
_SURFACE = {"red": 0.8, "green": 0.8, "blue": 0.8, "alpha": 1.0,
            "spec_amount": 0.0, "spec_shininess": 32.0, "emission": 0.0}
NODE_KEYS = {"type", "name", "params", "inputs", "pos", "disabled"}
OPTIONAL_NODE_KEYS = {"label", "thumbnail"}


def node_label(node):
    return node.get("label", "")


def node_thumbnail(node):
    return node.get("thumbnail", node["type"] in DEFAULT_THUMBNAIL_TYPES)
DEFAULT_SETTINGS = {
    "color": {
        "config": "ocio://cg-config-v4.0.0_aces-v2.0_ocio-v2.5",
        "working_space": "ACEScg",
        "display": "sRGB - Display",
        "view": "ACES 2.0",
    },
    "viewer": {"background": "black"},
}
# Viewer inputs and the A/B compare (lane L2 plan "2D Viewer parity", step V1). The viewer remembers
# up to nine inputs the way Nuke's Viewer node has nine input arrows: input N holds a node id, the
# active one is "A" and `doc["view"]` always equals it. The state lives in settings.viewer as four
# optional keys that appear together; a document that never used them has none, and one whose state
# is the default (input 1 = the viewed node, no B, "A only") is stored without them, so old and
# untouched documents keep their exact serialized form.
VIEWER_INPUT_COUNT = 9
COMPARE_MODES = ("A only", "B only", "wipe", "over", "under", "minus", "difference")
VIEWER_STATE_KEYS = ("inputs", "active", "b", "compare")
# Viewer gain, gamma, clipping warning and display choice (plan step V2). Display-only, like the
# exposure control it extends: they change the picture on screen and nothing the graph produces.
# One optional key, `look`, stored only when it differs from the default so old and untouched
# documents keep their exact serialized form. `display` is "Project view" (follow the project's
# default view) or the name of one of the config's views (validated against the config by the app,
# by name only here so core stays free of OCIO).
VIEWER_LOOK_DEFAULT = {"gain": 0.0, "gamma": 1.0, "zebra": False, "display": "Project view"}
VIEWER_GAIN_RANGE = (-10.0, 10.0)
VIEWER_GAMMA_RANGE = (0.2, 5.0)
# Region of interest, proxy and format masks (plan step V3). Three more optional viewer keys, each
# stored only when it differs from its default: `roi` {"on", "rect"} with the rectangle as fractions
# of the canvas [x0, y0, x1, y1] (top-left origin, so it survives a proxy or format change),
# `proxy` the viewer's evaluation tier, `masks` {"mask", "mode"}.
VIEWER_ROI_DEFAULT = {"on": False, "rect": [0.0, 0.0, 1.0, 1.0]}
VIEWER_PROXY_TIERS = (1, 2, 4, 8)
VIEWER_MASKS = ("format", "1.33", "1.66", "1.78", "1.85", "2.35", "2.40")
VIEWER_MASK_MODES = ("none", "lines", "half", "full")
VIEWER_MASKS_DEFAULT = {"mask": "format", "mode": "none"}
VIEWER_EXTRA_KEYS = ("look", "roi", "proxy", "masks")
SPECS = {
    # Read owns its own timeline-frame -> source-frame mapping (see docs/TIME_MODEL.md). "path" may
    # be a padded sequence pattern (plate.%04d.exr / plate.####.exr) or a still; "frame_offset"
    # shifts the source relative to the timeline, and "missing" decides what an absent frame does.
    "Read": {"inputs": [], "params": {"path": "", "colorspace": "Auto", "alpha_mode": "Auto", "layer": "", "subimage": 0,
                                      "frame_offset": 0, "missing": "error"}},
    "Constant": {"inputs": [], "params": {"width": 960, "height": 540, "red": 0.12, "green": 0.3, "blue": 0.6, "alpha": 1.0}},
    "Checker": {"inputs": [], "params": {"width": 960, "height": 540, "size": 64}},
    "Grade": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"exposure": 0.0, "multiply": 1.0, "offset": 0.0, "mix": 1.0}},
    "ColorCorrect": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"lift": 0.0, "gamma": 1.0, "gain": 1.0, "saturation": 1.0, "mix": 1.0}},
    "Blur": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"radius": 8.0, "mix": 1.0}},
    # Invert/Clamp/Multiply/Add/Gamma/Saturation are Nuke's Color-toolbar nodes that split a
    # single knob each out of Grade/ColorCorrect into its own scrub-friendly node. Multiply, Add
    # and Gamma deliberately reuse Grade's "multiply"/"offset" and ColorCorrect's "gamma" param
    # names (and their existing LIMITS) rather than inventing a parallel "value" name: the math is
    # the identical knob, just alone on its own node, exactly as lanes.md's ranked list frames it.
    "Invert": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"channels": "rgb", "mix": 1.0}},
    "Clamp": {"inputs": ["image"], "optional_inputs": ["mask"],
              "params": {"minimum": 0.0, "maximum": 1.0, "clamp_min": 1, "clamp_max": 1,
                        "channels": "rgb", "mix": 1.0}},
    "Multiply": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"multiply": 1.0, "channels": "rgb", "mix": 1.0}},
    "Add": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"offset": 0.0, "channels": "rgb", "mix": 1.0}},
    "Gamma": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"gamma": 1.0, "channels": "rgb", "mix": 1.0}},
    "Saturation": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"saturation": 1.0, "mix": 1.0}},
    # Erode/Dilate/Median/Sharpen/Glow/Mirror are the lane's group (c) Filter/Transform nodes.
    # Erode's own "erode_size" and Dilate's "dilate_size" share one box min/max kernel
    # (Evaluator._morph): a positive size erodes (shrinks) and a negative size dilates (grows) on
    # Erode; Dilate is "the positive twin" Nuke's toolbar offers as a separate node, so a positive
    # dilate_size grows instead. Distinct param names (not a shared "size") because LIMITS/CHOICES
    # are keyed by parameter name across every node and Checker's own "size" is an unrelated,
    # unsigned, integer pixel-grid spacing.
    "Erode": {"inputs": ["image"], "optional_inputs": ["mask"],
              "params": {"erode_size": 1.0, "channels": "rgba", "mix": 1.0}},
    "Dilate": {"inputs": ["image"], "optional_inputs": ["mask"],
               "params": {"dilate_size": 1.0, "channels": "rgba", "mix": 1.0}},
    "Median": {"inputs": ["image"], "optional_inputs": ["mask"],
               "params": {"median_size": 1.0, "channels": "rgba", "mix": 1.0}},
    "Sharpen": {"inputs": ["image"], "optional_inputs": ["mask"],
                "params": {"sharpen_amount": 0.5, "sharpen_size": 1.0, "channels": "rgb", "mix": 1.0}},
    # Glow's tint reuses the "red"/"green"/"blue" names Light3D and Relight already use for a
    # colour knob sharing their LIMITS, rather than inventing tint_red/tint_green/tint_blue.
    "Glow": {"inputs": ["image"], "optional_inputs": ["mask"],
             "params": {"glow_threshold": 1.0, "glow_size": 8.0, "brightness": 1.0,
                       "red": 1.0, "green": 1.0, "blue": 1.0, "channels": "rgb", "mix": 1.0}},
    # Soften (step 3a) is Nuke's Filter-menu Soften: a Gaussian-leaning sibling of Blur's box
    # filter. Its own "soften_size" name for the same reason as the other padded filters.
    "Soften": {"inputs": ["image"], "optional_inputs": ["mask"],
               "params": {"soften_size": 4.0, "channels": "rgba", "mix": 1.0}},
    # Exposure (step 3a) is Nuke's standalone Color-menu Exposure. Knob names follow the reference
    # guide (channels, blackpoint, gang, red/green/blue); Nuke's "mode" is `exposure_mode` here
    # because CHOICES is keyed globally by parameter name and "mode" already belongs to Tracker.
    "Exposure": {"inputs": ["image"], "optional_inputs": ["mask"],
                 "params": {"exposure_mode": "stops", "blackpoint": 0.0, "gang": 1, "red": 0.0,
                            "green": 0.0, "blue": 0.0, "channels": "rgb", "mix": 1.0}},
    # HueCorrect (step 4a) is Nuke's Color-menu HueCorrect reduced to the smallest honest model of
    # its per-hue curves: a fixed set of six hue anchors (red, yellow, green, cyan, blue, magenta,
    # 60 degrees apart, the same hue axis Nuke's curve editor uses), each with a saturation and a
    # luminance multiplier, smoothly interpolated between neighbours (docs/PARITY_2D.md).
    "HueCorrect": {"inputs": ["image"], "optional_inputs": ["mask"],
                   "params": {"sat_red": 1.0, "sat_yellow": 1.0, "sat_green": 1.0, "sat_cyan": 1.0, "sat_blue": 1.0, "sat_magenta": 1.0, "lum_red": 1.0, "lum_yellow": 1.0, "lum_green": 1.0, "lum_cyan": 1.0, "lum_blue": 1.0, "lum_magenta": 1.0,
                             "hue_shift": 0.0, "mix": 1.0}},
    # ColorMatrix (step 4a): a 3x3 RGB matrix as nine knobs, matrix_RC = row R, column C, so
    # out.r = matrix_00 * r + matrix_01 * g + matrix_02 * b. Defaults to the identity.
    "ColorMatrix": {"inputs": ["image"], "optional_inputs": ["mask"],
                    "params": {"matrix_00": 1.0, "matrix_01": 0.0, "matrix_02": 0.0, "matrix_10": 0.0, "matrix_11": 1.0, "matrix_12": 0.0, "matrix_20": 0.0, "matrix_21": 0.0, "matrix_22": 1.0,
                              "invert": 0, "mix": 1.0}},
    # Defocus (step 3b) is Nuke's disc blur without depth (ZDefocus stays missing: no depth
    # channel). "defocus" is the disc radius in pixels, "aspect" the disc's width / height.
    "Defocus": {"inputs": ["image"], "optional_inputs": ["mask"],
                "params": {"defocus": 6.0, "aspect": 1.0, "channels": "rgba", "mix": 1.0}},
    # DirBlur (step 3b) is Nuke's directional blur. blur_type: linear blurs along `angle` over
    # `length` pixels; zoom smears toward/away from (center_x, center_y) over `length` percent;
    # radial smears around it over a sweep of `angle` degrees. The centre is in canvas pixels
    # (default: the centre of the default 960x540 format).
    "DirBlur": {"inputs": ["image"], "optional_inputs": ["mask"],
                "params": {"blur_type": "linear", "angle": 0.0, "length": 20.0,
                           "center_x": 480.0, "center_y": 270.0, "channels": "rgba", "mix": 1.0}},
    # DropShadow (step 3b): the input's alpha, offset by `distance` along `angle` (degrees,
    # counter-clockwise from +x with y up; the default points down and to the right), blurred by
    # "shadow_size" (Nuke's `size`, renamed because "size" is a global LIMITS key with a 1 minimum
    # and a different meaning on Checker/Noise), tinted and put UNDER the input.
    "DropShadow": {"inputs": ["image"], "optional_inputs": ["mask"],
                   "params": {"angle": -45.0, "distance": 10.0, "shadow_size": 5.0, "opacity": 0.5,
                              "red": 0.0, "green": 0.0, "blue": 0.0, "mix": 1.0}},
    # EdgeBlur (step 5a): blur only along the matte's edge. "edgeblur_size" is the Gaussian's
    # pixel reach (Nuke's `size`, renamed because "size" is a global LIMITS key with a 1 minimum);
    # `edge_mult` scales the width of the band around the alpha edge the blur is confined to
    # (band = edge_mult * size pixels each side of the edge).
    "EdgeBlur": {"inputs": ["image"], "optional_inputs": ["mask"],
                 "params": {"edgeblur_size": 4.0, "edge_mult": 1.0, "channels": "rgba", "mix": 1.0}},
    # EdgeExtend (step 5a): dilates the unpremultiplied edge colour outward by `extend_size`
    # pixels from every pixel whose alpha is at least `extend_threshold`; alpha is unchanged and
    # the colour output is unpremultiplied (docs/PARITY_2D.md).
    "EdgeExtend": {"inputs": ["image"], "optional_inputs": ["mask"],
                   "params": {"extend_size": 3.0, "extend_threshold": 0.5, "mix": 1.0}},
    # LightWrap (step 5a): Nuke's two-input finishing node. Inputs are named fg and bg as in Nuke,
    # so a bypass passes the foreground (the first slot). "wrap_diffuse" is Nuke's `diffuse`,
    # renamed because "diffuse" is a 0..1 LIMITS key on Relight. red/green/blue are the constant
    # highlight colour, used when use_constant_highlight is on.
    "LightWrap": {"inputs": ["fg", "bg"], "optional_inputs": ["mask"],
                  "params": {"intensity": 1.0, "wrap_diffuse": 10.0, "fgblur": 1.0, "bgblur": 4.0,
                             "wrap_threshold": 0.0, "highlight_merge": "plus", "use_constant_highlight": 0,
                             "red": 1.0, "green": 1.0, "blue": 1.0, "mix": 1.0}},
    # Dither (step 5a): quantises to `bits` per channel with position-hashed triangular noise of
    # +/- `dither_amount` least-significant bits; `seed` picks the noise pattern.
    "Dither": {"inputs": ["image"], "optional_inputs": ["mask"],
               "params": {"bits": 8, "dither_amount": 1.0, "seed": 0, "channels": "rgb", "mix": 1.0}},
    # Grain (step 5b): synthetic film grain. Nuke's per-channel `size` and `intensity` knobs are
    # red_size/red_intensity etc. (Nuke's red_m is called red_intensity here); `seed` plus the
    # frame picks the pattern, so grain moves every frame and seed = -frame freezes it, as in Nuke.
    # `luminance_weighted` scales the grain by pixel luminance with `black` as the floor.
    "Grain": {"inputs": ["image"], "optional_inputs": ["mask"],
              "params": {"seed": 134, "red_size": 3.3, "green_size": 2.9, "blue_size": 2.5,
                         "red_intensity": 0.05, "green_intensity": 0.05, "blue_intensity": 0.05,
                         "luminance_weighted": 0, "black": 0.0, "mix": 1.0}},
    # Posterize (step 5b): `colors` levels per channel in the selected channels.
    "Posterize": {"inputs": ["image"], "optional_inputs": ["mask"],
                  "params": {"colors": 16, "channels": "rgb", "mix": 1.0}},
    # SoftClip (step 5b): Nuke's four `conversion` modes with softclip_min / softclip_max.
    "SoftClip": {"inputs": ["image"], "optional_inputs": ["mask"],
                 "params": {"conversion": "none", "softclip_min": 0.8, "softclip_max": 1.0, "mix": 1.0}},
    # HSVTool (step 5b): hue rotation, saturation and brightness adjustment limited to a hue,
    # saturation and brightness range each with a rolloff. Nuke's `huesrcs` pair is
    # hue_range_min/max, `satsrcs` is saturation_range_*, `brtsrcs` is brightness_range_*; Nuke's
    # `saturation`/`brightness` adjustments are sat_adjust/brt_adjust here because those names are
    # already other nodes' knobs. output_alpha writes the combined range weight into alpha.
    "HSVTool": {"inputs": ["image"], "optional_inputs": ["mask"],
                "params": {"hue_range_min": 0.0, "hue_range_max": 360.0, "hue_rolloff": 0.0, "hue_rotation": 0.0,
                           "saturation_range_min": 0.0, "saturation_range_max": 1.0, "saturation_rolloff": 0.0,
                           "sat_adjust": 0.0, "set_saturation": 0,
                           "brightness_range_min": 0.0, "brightness_range_max": 1.0, "brightness_rolloff": 0.0,
                           "brt_adjust": 0.0, "set_brightness": 0, "output_alpha": 0, "mix": 1.0}},
    # AddMix (step 5b): A is premultiplied, then merged `over` B (MERGE_LIKE_KINDS: bypass passes B).
    "AddMix": {"inputs": ["A", "B"], "optional_inputs": ["mask"], "params": {"mix": 1.0}},
    # Blend (step 5b): weighted average of up to eight inputs, one `weightN` per input. The first
    # two are required, so a bypass passes the first wired input (core.bypass_slot).
    "Blend": {"inputs": ["in0", "in1"],
              "optional_inputs": ["in2", "in3", "in4", "in5", "in6", "in7", "mask"],
              "params": {"weight0": 1.0, "weight1": 1.0, "weight2": 1.0, "weight3": 1.0, "weight4": 1.0,
                         "weight5": 1.0, "weight6": 1.0, "weight7": 1.0, "normalize": 1,
                         "channels": "rgba", "mix": 1.0}},
    # CopyRectangle (step 5b): copies the `area` box (area_x, area_y = top-left, area_r, area_t =
    # right and bottom edge, canvas pixels, rows counted from the top like Crop) from A over B.
    "CopyRectangle": {"inputs": ["A", "B"], "optional_inputs": ["mask"],
                      "params": {"channels": "rgba", "area_x": 512.0, "area_y": 389.0, "area_r": 1536.0,
                                 "area_t": 1167.0, "softness": 0.0, "mix": 1.0}},
    # Position: Nuke's integer-pixel move, `translate` as two ints. Pixels and data window move
    # together, nothing is resampled.
    "Position": {"inputs": ["image"], "params": {"translate_x": 0, "translate_y": 0}},
    # BlackOutside: black outside the data window, and the data window grows one pixel on each side
    # so a later filter has a black border to read instead of smearing the edge pixel.
    "BlackOutside": {"inputs": ["image"], "params": {}},
    # AdjustBBox: grow (or, negative, shrink) the data window by `numpixels` on every side without
    # moving pixels; `clip_to_format` then keeps it inside the display window.
    "AdjustBBox": {"inputs": ["image"], "params": {"numpixels": 0, "clip_to_format": 0}},
    "Mirror": {"inputs": ["image"], "optional_inputs": ["mask"],
               "params": {"flip_x": 0, "flip_y": 0, "mix": 1.0}},
    "Transform": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                                                 "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest", "mix": 1.0}},
    "Crop": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"x": 0, "y": 0, "width": 960, "height": 540, "mix": 1.0}},
    # Reformat is the lane's step 2c5 "real format model" (docs/PARITY_2D.md): unlike Transform/
    # Crop/Mirror it changes the *display* window itself, not just the data window -- see
    # `imaging.Evaluator._windowed_kernel`'s dedicated "Reformat" branch. The document-level named-
    # format registry the audit originally sketched ("stored in the document so a later node can
    # refer to a format by name") would touch files other lanes own, so this ships node-local
    # instead: `format` is a convenience preset that the Dispatcher's "set" op resolves into the
    # node's own width/height/pixel_aspect the moment it is chosen (`REFORMAT_FORMATS`, below) --
    # those three params are the only source of truth the kernel and the proxy-tier scaler
    # (`tiers.PIXEL_UNIT_PARAMS`) ever read, which is what keeps a named-format Reformat correct at
    # every playback tier. "to_box" reuses the same width/height/pixel_aspect fields as "Custom"
    # rather than inventing a parallel set of box_* knobs. pixel_aspect is carried for format-parity
    # but, like Retime's unused range-end knobs, is not consulted by the resample math, which works
    # in square pixels throughout this build.
    "Reformat": {"inputs": ["image"], "optional_inputs": ["mask"],
                 "params": {"reformat_type": "to_format", "format": "HD_1080",
                           "width": 1920, "height": 1080, "pixel_aspect": 1.0, "scale": 1.0,
                           "resize_type": "fit", "center": 1, "flip": 0, "flop": 0, "turn": 0,
                           "filter": "bilinear", "preserve_bbox": 0, "mix": 1.0}},
    # CornerPin2D: a projective (four-point) warp. Unlike Reformat it does not change the format --
    # like Transform/Tracker it only moves the data window, sharing their `_filter_window`/
    # `_filtered_pixels` dispatch and region-rule shape (`tiers._cornerpin_rule`). Default from/to
    # points are a 960x540 canvas's own corners, so a freshly created CornerPin is the identity
    # transform in `docs/PARITY_2D.md`'s own worked example. `direction` mirrors Nuke's "invert"
    # checkbox: "forward" warps the "from" quad onto the "to" quad; "inverse" swaps which quad is
    # the pre-warp side.
    "CornerPin": {"inputs": ["image"], "optional_inputs": ["mask"],
                  "params": {"from1_x": 0.0, "from1_y": 0.0, "from2_x": 960.0, "from2_y": 0.0,
                            "from3_x": 0.0, "from3_y": 540.0, "from4_x": 960.0, "from4_y": 540.0,
                            "to1_x": 0.0, "to1_y": 0.0, "to2_x": 960.0, "to2_y": 0.0,
                            "to3_x": 0.0, "to3_y": 540.0, "to4_x": 960.0, "to4_y": 540.0,
                            "direction": "forward", "filter": "bilinear", "mix": 1.0}},
    # STMap / IDistort / VectorBlur (step 5c): the 2D side of the control loop. Each takes the image,
    # an optional `uv` image (a map or a vector field) and an optional mask. `uv_layer` names a
    # layer of the `uv` input when it is wired, otherwise of the image input itself, so a
    # multichannel Render3D or EXR Read can drive the warp on its own; empty means "use the uv
    # input's beauty". u_channel/v_channel pick which channels of that raster are u and v.
    "STMap": {"inputs": ["image"], "optional_inputs": ["uv", "mask"],
              "params": {"uv_layer": "", "u_channel": "R", "v_channel": "G", "filter": "bilinear",
                         "uv_outside": "black", "mix": 1.0}},
    "IDistort": {"inputs": ["image"], "optional_inputs": ["uv", "mask"],
                 "params": {"uv_layer": "", "u_channel": "R", "v_channel": "G",
                            "uv_scale_x": 1.0, "uv_scale_y": 1.0, "uv_offset_x": 0.0, "uv_offset_y": 0.0,
                            "filter": "bilinear", "mix": 1.0}},
    "VectorBlur": {"inputs": ["image"], "optional_inputs": ["uv", "mask"],
                   "params": {"uv_layer": "", "u_channel": "R", "v_channel": "G",
                              "vector_scale": 1.0, "vector_offset": 0.0, "vector_method": "forward",
                              "vector_alpha": "none", "max_length": 100.0, "mix": 1.0}},
    # Shuffle's `layer` (step 5c) names one of the input's named layers (Raster.layers); empty or
    # "rgba" shuffles the input's own channels.
    "Shuffle": {"inputs": ["image"], "params": {"red_from": "R", "green_from": "G", "blue_from": "B", "alpha_from": "A", "layer": ""}},
    # Two-input explicit channel routing. Separate from Shuffle (one input, kept unchanged) because
    # CHOICES is keyed by parameter name globally — sharing "red_from" would force one option list
    # on both nodes. Naming a B.* source with B unwired is an error, never a silent black channel.
    "ChannelShuffle": {"inputs": ["A"], "optional_inputs": ["B"],
                       "params": {"out_red": "A.r", "out_green": "A.g", "out_blue": "A.b", "out_alpha": "A.a"}},
    # Roto is a generator: it states its own format rather than inheriting one from an image input
    # and then quietly disagreeing with it. Shapes live in document["node_data"], not in params —
    # see docs/ROTO_TRACKING.md.
    "Roto": {"inputs": [], "params": {"width": 960, "height": 540, "invert": 0}},
    # Ramp/Radial/Rectangle/Noise/Text are the lane's group (c2) Draw-menu generators: like
    # Constant/Checker/Roto they state their own format (width/height) rather than inheriting one,
    # but unlike those three they also take an optional "image" input the shape is composited over
    # (DRAW_KINDS, below) -- Nuke's own Draw-node convention. A wired image must already be at the
    # node's own width/height (M0); there is no resampling. mask + mix follow the same contract as
    # every other gated node in this file.
    "Ramp": {"inputs": [], "optional_inputs": ["image", "mask"],
             "params": {"width": 960, "height": 540,
                       "p0_x": 0.0, "p0_y": 0.0, "p1_x": 959.0, "p1_y": 0.0,
                       "color0_red": 0.0, "color0_green": 0.0, "color0_blue": 0.0, "color0_alpha": 1.0,
                       "color1_red": 1.0, "color1_green": 1.0, "color1_blue": 1.0, "color1_alpha": 1.0,
                       "mix": 1.0}},
    "Radial": {"inputs": [], "optional_inputs": ["image", "mask"],
               "params": {"width": 960, "height": 540,
                         "box_x": 330.0, "box_y": 145.0, "box_width": 300.0, "box_height": 250.0,
                         "softness": 0.5, "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0,
                         "mix": 1.0}},
    "Rectangle": {"inputs": [], "optional_inputs": ["image", "mask"],
                  "params": {"width": 960, "height": 540,
                            "box_x": 330.0, "box_y": 145.0, "box_width": 300.0, "box_height": 250.0,
                            "softness": 0.0, "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0,
                            "mix": 1.0}},
    # Grid (lane L2 step 4c) is Nuke's Draw Grid: vertical and horizontal lines every `spacing_*`
    # pixels, starting at `grid_offset_*`, `line_width` pixels wide. `number_x`/`number_y` above zero
    # switch a direction to "this many lines across the format" (spacing = format size / number),
    # Nuke's own number-or-size choice folded into one knob pair. The names are distinct from
    # every other node's because LIMITS is keyed globally by parameter name.
    "Grid": {"inputs": [], "optional_inputs": ["image", "mask"],
             "params": {"width": 960, "height": 540,
                       "spacing_x": 64.0, "spacing_y": 64.0, "number_x": 0, "number_y": 0,
                       "grid_offset_x": 0.0, "grid_offset_y": 0.0, "line_width": 1.0,
                       "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0, "mix": 1.0}},
    # NoOp: Nuke's passthrough that, unlike Dot (a wire reroute with no properties), is a real node
    # with a properties panel and a place for user notes. Nothing reads `note`; bypassed or not, the
    # input passes through untouched (`bypass_slot`'s first-input rule).
    "NoOp": {"inputs": ["image"], "params": {"note": ""}},
    # Noise's "size" is Nuke's own knob name for the feature size in pixels; z_slice/octaves/
    # lacunarity/gain/gamma are Nuke's own fractal-noise knobs too. seed is not a Nuke Noise knob
    # (Nuke reseeds from z_slice alone); it is added because the lane brief requires two runs with
    # the same seed to be identical and two different seeds to differ, which a z_slice-only
    # generator cannot promise without moving the slice.
    "Noise": {"inputs": [], "optional_inputs": ["image", "mask"],
              "params": {"width": 960, "height": 540, "size": 64.0, "z_slice": 0.0, "octaves": 4,
                        "lacunarity": 2.0, "gain": 0.5, "gamma": 1.0, "seed": 0, "mix": 1.0}},
    # Text renders through Qt's own text rasteriser offscreen (QPainter/QFont on a QImage), so it
    # needs no new dependency. "font" is a family name looked up through Qt's font database, not a
    # file path: whether a given family actually renders depends on what is installed on the
    # machine running NodeBased, so an artist who relies on a specific typeface should confirm it
    # renders the same on every machine that will open the comp (docs/PARITY_2D.md records this).
    "Text": {"inputs": [], "optional_inputs": ["image", "mask"],
             "params": {"width": 960, "height": 540, "message": "Text", "font": "", "font_size": 48.0,
                       "box_x": 40.0, "box_y": 40.0, "box_width": 880.0, "box_height": 460.0,
                       "justify": "left", "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0,
                       "mix": 1.0}},
    # Tracker is a Transform whose transform is solved from tracks in node_data instead of typed
    # in. It honours the same optional-mask + mix contract as the image filters.
    "Tracker": {"inputs": ["image"], "optional_inputs": ["mask"],
                "params": {"reference_frame": 1, "mode": "match_move", "apply_translate": 1,
                           "apply_rotate": 1, "apply_scale": 1, "filter": "bilinear", "mix": 1.0}},
    # Merge's optional mask gates the merge per pixel exactly as `mix` gates it globally: where
    # mask.a is 0 the output is B untouched. Added in v11; see `upgrade_document`.
    "Merge": {"inputs": ["A", "B"], "optional_inputs": ["mask"],
              "params": {"operation": "over", "mix": 1.0}},
    # Dissolve/Keymix/Copy/ChannelMerge are Nuke's other Merge-toolbar two-input nodes; all four
    # share Merge's own mask + mix contract (Foundry's Merge2 base class every one of them
    # inherits from) and MERGE_LIKE_KINDS bypass/windowing convention.
    "Dissolve": {"inputs": ["A", "B"], "optional_inputs": ["mask"], "params": {"which": 0.0, "mix": 1.0}},
    "Keymix": {"inputs": ["A", "B"], "optional_inputs": ["mask"], "params": {"invert_mask": 0, "mix": 1.0}},
    # Copy replaces named channels of B with channels from A; "none" leaves that output channel
    # as B's own. Named copy_* (not red_from/green_from/...) because those names are already
    # Shuffle's CHOICES with a different option list (routes from A, B or a constant).
    "Copy": {"inputs": ["A", "B"], "optional_inputs": ["mask"],
             "params": {"copy_red": "none", "copy_green": "none", "copy_blue": "none",
                       "copy_alpha": "none", "mix": 1.0}},
    "ChannelMerge": {"inputs": ["A", "B"], "optional_inputs": ["mask"],
                     "params": {"a_channel": "A.a", "b_channel": "B.a", "out_channel": "A",
                               "operation": "over", "mix": 1.0}},
    # Keyer/HueKeyer/Difference are the lane's group (c3) Keyer-menu nodes. Keyer keys a chosen
    # per-pixel quantity through a four-point range ramp (0 below range_a, ramping up to 1 between
    # range_a and range_b, 1 through range_c, ramping down to 0 between range_c and range_d, 0
    # above range_d) into alpha; it honours the single-image mask + mix contract like Saturation.
    "Keyer": {"inputs": ["image"], "optional_inputs": ["mask"],
              "params": {"keyer_operation": "luminance", "range_a": 0.0, "range_b": 0.0,
                        "range_c": 1.0, "range_d": 1.0, "invert": 0, "mix": 1.0}},
    # HueKeyer: a hue range with softness (hue_center/hue_width/hue_softness, degrees) plus a hard
    # saturation range (sat_min/sat_max). Nuke's own HueKeyer knobs are collapsed to these numeric
    # fields -- see docs/PARITY_2D.md's note on the simplification.
    "HueKeyer": {"inputs": ["image"], "optional_inputs": ["mask"],
                 "params": {"hue_center": 0.0, "hue_width": 30.0, "hue_softness": 15.0,
                           "sat_min": 0.0, "sat_max": 1.0, "invert": 0, "mix": 1.0}},
    # Difference: Nuke's two-input colour-difference keyer. Reuses Grade's "offset" and
    # ColorCorrect's "gain" param names/LIMITS rather than inventing new ones (same convention as
    # Multiply/Add/Gamma reusing Grade's own knobs). MERGE_LIKE_KINDS: bypass passes B.
    "Difference": {"inputs": ["A", "B"], "optional_inputs": ["mask"],
                   "params": {"offset": 0.0, "gain": 1.0, "mix": 1.0}},
    # TimeOffset/FrameHold/Retime are the lane's group (c4) Time-menu nodes: each is a producer
    # with its own time mapping (`nodebased.imaging._time_remap_frame`, docs/TIME_MODEL.md) rather
    # than a pixel kernel, so none takes a mask or mix -- Nuke's own Time-menu nodes don't have
    # them either. TimeOffset shifts the input by a frame count; "reverse" flips which direction
    # the offset applies. FrameHold holds on "first_frame" (increment 0, Nuke's own default) or
    # steps forward every "increment" frames. Retime is the simplified nearest-frame version: a
    # frame maps from the output range onto the input range, anchored at each range's start and
    # scaled by "speed" -- see docs/PARITY_2D.md's note on the simplification (no frame blending).
    "TimeOffset": {"inputs": ["image"], "params": {"time_offset": 0, "reverse": 0}},
    "FrameHold": {"inputs": ["image"], "params": {"first_frame": 1, "increment": 0}},
    "Retime": {"inputs": ["image"],
              "params": {"input_range_start": 0, "input_range_end": 100,
                        "output_range_start": 0, "output_range_end": 100, "speed": 1.0}},
    # TimeClip/FrameRange/AppendClip (step 4b) are the same kind of producer with its own time
    # mapping (no mask, no mix). TimeClip: the input is evaluated at `frame - time_offset`; frames
    # outside [first, last] (when "frame_range_type" is "custom"; "all" ignores the range) follow
    # "before"/"after" (hold, loop, bounce, black). FrameRange is the same clamp without the offset,
    # under Nuke's own knob names (first_frame/last_frame): the document has no per-branch frame
    # range, so it presents the range by clamping rather than by metadata. AppendClip plays up to
    # eight clips (clip0..clip7, as Scene3D's object slots) head to tail from "first_frame"; each
    # clip's length is its FrameRange/TimeClip range when that is directly upstream, else its
    # "length<i>" knob (0 skips the clip); "dissolve" frames cross-fade consecutive clips.
    "TimeClip": {"inputs": ["image"],
                 "params": {"first": 1, "last": 100, "frame_range_type": "custom",
                            "before": "hold", "after": "hold", "time_offset": 0}},
    "FrameRange": {"inputs": ["image"],
                   "params": {"first_frame": 1, "last_frame": 100, "before": "hold", "after": "hold"}},
    "AppendClip": {"inputs": [], "optional_inputs": [f"clip{i}" for i in range(8)],
                   "params": {"first_frame": 1, "dissolve": 0, **{f"length{i}": 100 for i in range(8)}}},
    "Premult": {"inputs": ["image"], "params": {}},
    "Unpremult": {"inputs": ["image"], "params": {}},
    "Dot": {"inputs": ["input"], "params": {}},
    "Switch": {"inputs": ["0", "1"], "params": {"which": 0}},
    "Viewer": {"inputs": ["image"], "params": {}},
    # Write is where real image output lives, as in Nuke: the node states the destination and the
    # format, and rendering it is an explicit action rather than a side effect of looking at a
    # frame. It passes its input through unchanged, so a Write parked mid-branch never alters the
    # comp downstream of it. "file_type" Auto takes the extension on "path" at its word.
    "Write": {"inputs": ["image"], "params": {"path": "", "file_type": "Auto", "bit_depth": "half", "bundle": 0}},
    # ReadBundle (step 5c): reads a diffusion or transform model's output image for the graph's
    # frame and refuses it unless the bundle manifest Write wrote for that frame agrees on frame and
    # size (nodebased/bundle.py). `path` and `bundle` may be padded patterns (render.%04d.png,
    # render.%04d.bundle.json).
    "ReadBundle": {"inputs": [], "params": {"path": "", "bundle": "", "colorspace": "Auto", "alpha_mode": "Auto"}},
    # Bounded 3D foundation. Geometry/camera nodes are typed scene data; Render3D is the
    # image-producing bridge, so ordinary Grade/Merge/Write nodes can consume its output.
    # Every 3D parameter has its own name (tx, card_width, ...) because LIMITS and CHOICES are
    # keyed by parameter name across all nodes: reusing "x" or "width" would inherit pixel bounds.
    # Axis3D is a Nuke-style pure transform: it parents whatever typed 3D value is wired into it
    # (geometry, light or a whole scene) and produces a scene, so chaining Axis3D nodes composes
    # transforms in order via ordinary scene nesting (see scene3d.scene_from_node). With nothing
    # wired it produces an empty scene.
    "Axis3D": {"inputs": [], "optional_inputs": ["object"], "params": dict(_XFORM)},
    # TransformGeo3D bakes its transform directly into the incoming geometry's own vertices and
    # normals (nodebased.scene3d.transform_geometry), unlike Axis3D which only ever adds another
    # parent matrix. This lets a modeling chain flatten a transform before further edits.
    "TransformGeo3D": {"inputs": ["geo"], "params": dict(_XFORM)},
    # MergeGeo3D concatenates up to eight geometries into ONE, baking each input's world transform
    # (and then this node's own transform) into the vertices. Same eight optional slot pattern as
    # Scene3D, but the slots take geometry only and the output is geometry, so it can feed
    # TransformGeo3D, Normals3D, DisplaceGeo3D, WriteGeo3D or a Scene3D slot.
    "MergeGeo3D": {"inputs": [], "optional_inputs": [f"geo{i}" for i in range(8)], "params": dict(_XFORM)},
    # Normals3D: "normals_mode" (CHOICES is keyed by parameter name across all nodes and "mode"
    # already belongs to Tracker) picks unchanged / recompute / flip / unify.
    "Normals3D": {"inputs": ["geo"], "params": {"normals_mode": "recompute", "flip_winding": 0}},
    # DisplaceGeo3D moves each vertex along its normal by displace_scale * (channel of the image
    # at the vertex UV) + displace_offset. Distinct names because LIMITS is keyed globally and
    # "scale"/"offset" already carry 2D bounds.
    "DisplaceGeo3D": {"inputs": ["geo"], "optional_inputs": ["image"],
                      "params": {"displace_scale": 1.0, "displace_offset": 0.0,
                                 "displace_channel": "luminance", "recompute_normals": 1}},
    "Card3D": {"inputs": [], "optional_inputs": ["image"],
               "params": {"card_width": 2.0, "card_height": 2.0, "rows": 1, "columns": 1,
                          **_XFORM, **_SURFACE}},
    "Cube3D": {"inputs": [], "optional_inputs": ["image"],
               "params": {"cube_size": 2.0, **_XFORM, **_SURFACE}},
    # "segments" is kept for old documents (upgrade_document derives rows/columns from it so
    # they render byte-identically); new Sphere3D nodes are driven by rows/columns alone.
    "Sphere3D": {"inputs": [], "optional_inputs": ["image"],
                 "params": {"sphere_radius": 1.0, "segments": 32, "rows": 16, "columns": 32,
                           **_XFORM, **_SURFACE}},
    # Cylinder3D: a new geometry primitive (lane L3 step 2). Axis along +Y, rows are height
    # segments, columns are radial segments (Nuke's own Cylinder knob names), "cyl_caps"
    # chooses flat end caps or an open tube.
    "Cylinder3D": {"inputs": [], "optional_inputs": ["image"],
                   "params": {"cyl_radius": 1.0, "cyl_height": 2.0, "rows": 1, "columns": 24,
                             "cyl_caps": "closed", **_XFORM, **_SURFACE}},
    # ParticleEmitter3D is a deterministic emitter (nodebased/particles.py, docs/SIMULATION.md). It
    # outputs "particles", which Scene3D/Axis3D slots accept, and needs no wired input: with none it
    # is a point emitter at its own transform. "geo" supplies the points, surface or volume to emit
    # from. Knob names follow Nuke's ParticleEmitter where it has one and Houdini's POP Source
    # otherwise; every name is distinct because LIMITS and CHOICES are keyed globally.
    "ParticleEmitter3D": {"inputs": [], "optional_inputs": ["geo"], "params": {
        "emit_from": "point", "emit_rate": 100.0, "emit_rate_unit": "per_frame", "start_frame": 1,
        "life": 24.0, "life_variance": 0.0, "emit_speed": 1.0, "speed_variance": 0.0,
        "emit_dir_x": 0.0, "emit_dir_y": 1.0, "emit_dir_z": 0.0, "direction_from_normals": 0,
        "spread": 0.0, "particle_size": 0.05, "size_variance": 0.0,
        "red": 1.0, "green": 1.0, "blue": 1.0, "alpha": 1.0,
        "seed": 0, "substeps": 1, "max_particles": 1000000, **_XFORM}},
    # ParticleCache3D solves the particles wired into it through the disk-backed simcache: every
    # frame is a checkpoint, scrubbing back never re-solves, and the two budgets bound the memory
    # tier and the disk tier for this node.
    "ParticleCache3D": {"inputs": ["particles"],
                        "params": {"cache_memory_mb": 256, "cache_disk_mb": 2048}},
    # Force nodes (step 2b) take a particle set in and out so they chain like Nuke's; each adds an
    # acceleration (units per frame squared, like emit_speed's units per frame) to the particles it
    # affects: `probability` of them (seeded per particle id), between `from_frame` and `to_frame`.
    "ParticleGravity3D": {"inputs": ["particles"], "params": {
        "gravity_x": 0.0, "gravity_y": -1.0, "gravity_z": 0.0, "strength": 0.02, **_FORCE}},
    "ParticleDrag3D": {"inputs": ["particles"], "params": {
        "drag": 0.05, "drag_quadratic": 0.0, **_FORCE}},
    "ParticleWind3D": {"inputs": ["particles"], "params": {
        "wind_x": 1.0, "wind_y": 0.0, "wind_z": 0.0, "strength": 0.02, "wind_gust": 0.0,
        "wind_gust_rate": 0.25, **_FORCE}},
    "ParticleTurbulence3D": {"inputs": ["particles"], "params": {
        "turb_mode": "curl", "turb_size": 1.0, "strength": 0.02, "octaves": 2, **_FORCE}},
    # ParticleBounce3D (step 2c) collides the particles against the geometry (or Scene3D) wired into
    # "geometry", sampled once at the emitter's start frame. `bounce` is the restitution, `friction` a
    # Coulomb coefficient, `kill_on_collision` removes a particle on contact. It chains like a force.
    "ParticleBounce3D": {"inputs": ["particles"], "optional_inputs": ["geometry"], "params": {
        "bounce": 0.6, "friction": 0.1, "kill_on_collision": 0, **_FORCE}},
    # ParticleRender3D (step 2c) chooses how the particles it passes on are drawn. It is a node of its
    # own rather than a knob on the emitter because a drawing choice must not change the run identity
    # (and so must not re-solve a cache): it sits after the forces and any ParticleCache3D.
    "ParticleRender3D": {"inputs": ["particles"], "optional_inputs": ["image"], "params": {
        "representation": "points", "size_scale": 1.0}},
    "ReadSplat3D": {"inputs": [], "params": {
        "splat_path": "", "splat_orientation": "as_authored", "splat_colorspace": "srgb",
        "splat_sh_degree": 3, "splat_opacity": 1.0, "splat_scale": 1.0, "splat_relight": 0.0,
        "splat_shadow_catch": 0.0, "splat_cast_shadows": "on", "splat_specular": 0.0, "splat_normal_smoothing": 0,
        # De-lighting (nodebased.intrinsics, docs/SPLAT_RELIGHTING.md): an offline fit that divides the
        # capture's own lighting out of its colour. Off keeps a document exactly as it was.
        "splat_delight": "off", "splat_delight_iterations": 12, "splat_delight_smoothness": 0.5,
        "splat_delight_light_order": 1, "splat_use_intrinsics": "on",
        **_XFORM}},
    "ReadAlembic3D": {"inputs": [], "params": {"abc_path": "", "abc_root": "/"}},
    "ReadAlembicCamera3D": {"inputs": [], "params": {"abc_path": "", "abc_camera": ""}},
    "ReadUSD3D": {"inputs": [], "params": {"usd_path": "", "usd_root": "/"}},
    "ReadUSDCamera3D": {"inputs": [], "params": {"usd_path": "", "usd_camera": ""}},
    "ReadGLTF3D": {"inputs": [], "params": {"gltf_path": "", "gltf_root": ""}},
    "ReadGeo3D": {"inputs": [], "optional_inputs": ["image"],
                  "params": {"geo_path": "", **_XFORM, **_SURFACE}},
    "Light3D": {"inputs": [], "params": {"light_type": "Directional", "tx": 2.0, "ty": 4.0, "tz": 3.0,
                                         "target_x": 0.0, "target_y": 0.0, "target_z": 0.0,
                                         "red": 1.0, "green": 1.0, "blue": 1.0, "intensity": 1.0, "shadows": "off",
                                         "cone_angle": 30.0, "cone_penumbra_angle": 5.0,
                                         "cone_falloff": 1.0, "falloff_type": "No falloff",
                                         "shadow_bias": 0.001, "shadow_blur": 0.0, "shadow_samples": 1}},
    "Camera3D": {"inputs": [], "params": {"tx": 0.0, "ty": 0.0, "tz": 5.0, "roll": 0.0,
                                          "target_x": 0.0, "target_y": 0.0, "target_z": 0.0,
                                          "focal": filmback.DEFAULT_FOCAL,
                                          "haperture": filmback.DEFAULT_HAPERTURE,
                                          "vaperture": filmback.DEFAULT_VAPERTURE,
                                          "near": 0.1, "far": 1000.0}},
    "Project3D": {"inputs": ["image", "camera", "geometry"],
                  "params": {"project_outside": "transparent", "project_backfaces": "project", "project_occlusion": "off"}},
    "WriteGeo3D": {"inputs": ["scene"], "params": {"geo_write_path": ""}},
    # WriteSplat3D writes the scene's splats (world transforms baked) to a 3DGS .ply on request.
    "WriteSplat3D": {"inputs": ["scene"], "params": {"splat_write_path": "", "splat_write_overwrite": 0}},
    "Scene3D": {"inputs": [], "optional_inputs": [f"object{i}" for i in range(8)], "params": dict(_XFORM)},
    "Relight": {"inputs": ["image"], "optional_inputs": ["camera"] + [f"light{i}" for i in range(8)],
                "params": {"red": 0.8, "green": 0.8, "blue": 0.8,
                           "diffuse": 1.0, "specular": 1.0, "mix": 1.0, "use_intrinsics": "on"}},
    "Render3D": {"inputs": ["scene", "camera"],
                 "params": {"width": 960, "height": 540, "red": 0.0, "green": 0.0, "blue": 0.0,
                            "alpha": 0.0, "ambient": 0.1, "samples": 2, "render_output": "rgba", "render_backend": "cpu", "render_mode": "raster",
                            "passes": "beauty,normals,depth"}},
}

# Plume3D is a synthetic smoke volume (scene3d.analytic_plume) for demos and tests until the fluid solver
# and the VDB reader land: a deterministic plume of the given grid resolution and seed under its own
# transform. It outputs "volume", which Scene3D and Axis3D slots accept and Render3D raymarches.
SPECS["Plume3D"] = {"inputs": [], "params": {"plume_resolution": 32, "plume_seed": 0, **_XFORM}}

# Render3D's volume knobs (docs/FLUIDS_SPIKE.md). Houdini Pyro's names where they exist: Density scale,
# Shadow density, Scattering, Absorption, Smoke color. `volumes` switches the raymarch on (off: the
# scene's volumes are ignored by every backend), `volume_fps` turns velocities into per-frame motion
# vectors and `volume_depth_threshold` is the scaled density at which the depth pass sees the smoke.
_VOLUME_RENDER_DEFAULTS = {
    "volumes": "on", "volume_step_size": 0.05, "volume_density_scale": 1.0, "volume_shadow_density": 1.0,
    "volume_shadow_steps": 16, "volume_scattering": 1.0, "volume_absorption": 0.2,
    "volume_red": 1.0, "volume_green": 1.0, "volume_blue": 1.0,
    "volume_fps": 24.0, "volume_depth_threshold": 0.1}
SPECS["Render3D"]["params"].update(_VOLUME_RENDER_DEFAULTS)


def builtin_formats():
    """The registry every document starts with: `REFORMAT_FORMATS` as named window records."""
    return {name: {"width": w, "height": h, "pixel_aspect": pa}
            for name, (w, h, pa) in REFORMAT_FORMATS.items()}


def document_formats(doc):
    """The document's named formats. A document that predates the registry (or a hand-built dict
    that never went through `upgrade_document`) falls back to the built-in list, so a lookup never
    depends on whether the section has been written yet."""
    settings = doc.get("settings") if isinstance(doc, dict) else None
    formats = settings.get("formats") if isinstance(settings, dict) else None
    return formats if isinstance(formats, dict) else builtin_formats()


def _resolve_reformat_format(params, doc=None):
    """A named-format pick resolves into `params["width"/"height"/"pixel_aspect"]` right here, so
    those three stay the only pixel-unit state the kernel and `tiers.scale_params` ever read (see
    the SPECS["Reformat"] comment). The document registry (`settings.formats`) is consulted first
    and the node-local built-in list second, so two Reformats naming the same format always mean
    the same window. Called from "create" (so a document that names a format directly, e.g. an
    agent-authored one, does not need a follow-up "set"), "set" (so changing the dropdown later has
    the same effect) and the registry edits (`_edit_format`), which re-resolve every node that
    names the edited entry. "Custom", or a name neither list knows, leaves width/height/
    pixel_aspect untouched.
    """
    name = params.get("format")
    entry = document_formats(doc).get(name) if doc is not None else None
    if entry is not None:
        params["width"], params["height"], params["pixel_aspect"] = (
            entry["width"], entry["height"], entry["pixel_aspect"])
        return
    preset = REFORMAT_FORMATS.get(name)
    if preset is not None:
        params["width"], params["height"], params["pixel_aspect"] = preset


FORMAT_NAME_LIMIT = 64
FORMAT_COUNT_LIMIT = 256


def _validate_format_entry(name, entry):
    if not isinstance(name, str) or not name.strip() or len(name) > FORMAT_NAME_LIMIT:
        raise ValueError(f"format names must be 1 to {FORMAT_NAME_LIMIT} characters")
    if name == "Custom":
        raise ValueError('"Custom" is reserved for a Reformat that states its own size')
    if not isinstance(entry, dict) or set(entry) != {"width", "height", "pixel_aspect"}:
        raise ValueError(f"format {name!r} must define width, height and pixel_aspect")
    for field in ("width", "height", "pixel_aspect"):
        value = entry[field]
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError(f"format {name!r}: {field} must be a finite number")
        if field != "pixel_aspect" and type(value) is not int:
            raise ValueError(f"format {name!r}: {field} must be an integer")
        lo, hi = LIMITS[field]
        if not lo <= value <= hi:
            raise ValueError(f"format {name!r}: {field} must be between {lo} and {hi}")


def _edit_format(doc, cmd):
    """The "format" op: edit the document's named-format registry (`settings.formats`).

    action "set" adds or updates `name`; "rename" moves `name` to `new_name`; "delete" removes it.
    Reformat nodes are edited in the same command so the registry stays the one place a format's
    meaning lives: a set re-resolves every node naming the entry, a rename rewrites their `format`
    param, and a delete turns them into "Custom" with the window they last had.
    """
    action = cmd.get("action")
    name = cmd.get("name")
    settings = doc["settings"]
    formats = settings.setdefault("formats", builtin_formats())
    nodes = [n for n in doc["nodes"].values() if n["type"] == "Reformat"]
    if action == "set":
        entry = {"width": cmd.get("width"), "height": cmd.get("height"),
                 "pixel_aspect": cmd.get("pixel_aspect", 1.0)}
        _validate_format_entry(name, entry)
        if name not in formats and len(formats) >= FORMAT_COUNT_LIMIT:
            raise ValueError(f"at most {FORMAT_COUNT_LIMIT} formats")
        formats[name] = entry
        for node in nodes:
            if node["params"]["format"] == name:
                _resolve_reformat_format(node["params"], doc)
    elif action == "rename":
        new_name = cmd.get("new_name")
        if name not in formats:
            raise ValueError(f"unknown format {name!r}")
        _validate_format_entry(new_name, formats[name])
        if new_name in formats and new_name != name:
            raise ValueError(f"format {new_name!r} already exists")
        settings["formats"] = {(new_name if key == name else key): value for key, value in formats.items()}
        for node in nodes:
            if node["params"]["format"] == name:
                node["params"]["format"] = new_name
    elif action == "delete":
        if name not in formats:
            raise ValueError(f"unknown format {name!r}")
        del formats[name]
        for node in nodes:
            if node["params"]["format"] == name:
                node["params"]["format"] = "Custom"
    else:
        raise ValueError("format: action must be set, rename or delete")
    return copy.deepcopy(settings["formats"])


def bypass_slot(node):
    """The one input slot a bypassed (disabled) node passes through, or None.

    Every evaluation path asks here, so the traversal, the cache digests and the pixels cannot
    disagree about what a bypassed node is. A Merge passes B, its background, as in Nuke: bypassing
    the merge removes what was laid over the main pipe rather than the pipe itself. With B unwired
    it passes A. Project3D passes its geometry. MergeGeo3D passes its first wired geometry. Axis3D passes its object (its only slot, but
    optional, so it is not in SPECS["Axis3D"]["inputs"]). Everything else passes its first declared
    input.
    """
    kind, inputs = node["type"], node["inputs"]
    if kind == "Project3D":
        return "geometry"
    if kind == "Axis3D":
        return "object"
    if kind == "ParticleEmitter3D":
        return "geo"   # the emission geometry (optional, so not in SPECS inputs); None when unwired
    if kind == "MergeGeo3D":
        # The first wired geometry slot; with none wired, geo0 (the bypass is then an empty geometry).
        return next((slot for slot in SPECS[kind]["optional_inputs"] if inputs.get(slot) is not None), "geo0")
    if kind == "AppendClip":
        # The first wired clip; with none wired, clip0 (the evaluator then reports the empty node).
        return next((slot for slot in SPECS[kind]["optional_inputs"] if inputs.get(slot) is not None), "clip0")
    if kind == "Blend":
        # No B input: the first wired numbered input (in0 when none is wired).
        return next((slot for slot in SPECS[kind]["inputs"] + SPECS[kind]["optional_inputs"][:-1]
                     if inputs.get(slot) is not None), "in0")
    if kind in MERGE_LIKE_KINDS:
        return "B" if inputs.get("B") is not None or inputs.get("A") is None else "A"
    if kind in DRAW_KINDS:
        return "image"
    slots = SPECS[kind]["inputs"]
    return slots[0] if slots else None


OUTPUT_TYPES = {kind: "image" for kind in SPECS}
GEOMETRY_TYPES = ("Card3D", "Cube3D", "Sphere3D", "Cylinder3D", "ReadGeo3D")
OUTPUT_TYPES.update({kind: "geometry" for kind in GEOMETRY_TYPES})
OUTPUT_TYPES.update({"ReadSplat3D": "scene", "ReadAlembic3D": "scene", "ReadAlembicCamera3D": "camera", "ReadUSD3D": "scene", "ReadUSDCamera3D": "camera", "ReadGLTF3D": "scene", "Light3D": "light", "Camera3D": "camera", "Scene3D": "scene", "Project3D": "scene", "WriteGeo3D": "scene", "WriteSplat3D": "scene", "Render3D": "image", "Axis3D": "scene", "TransformGeo3D": "geometry",
                    "MergeGeo3D": "geometry", "Normals3D": "geometry", "DisplaceGeo3D": "geometry",
                    "ParticleEmitter3D": "particles", "ParticleCache3D": "particles",
                    "ParticleGravity3D": "particles", "ParticleDrag3D": "particles",
                    "ParticleWind3D": "particles", "ParticleTurbulence3D": "particles",
                    "ParticleBounce3D": "particles", "ParticleRender3D": "particles"})
# A slot accepts a tuple of value types. Scene3D members may be geometry, lights or whole scenes
# (nesting is the hierarchy: a child scene inherits its parent's transform).
INPUT_TYPES = {"image": ("image",), "scene": ("scene",), "camera": ("camera",),
               "geometry": ("geometry", "scene"),
               # Axis3D's single slot accepts the same members a Scene3D object slot does.
               "object": ("geometry", "light", "scene", "particles", "volume"),
               # TransformGeo3D bakes vertices directly, so it takes one geometry, never a scene.
               "geo": ("geometry",)}
INPUT_TYPES.update({f"object{i}": ("geometry", "light", "scene", "particles", "volume") for i in range(8)})
INPUT_TYPES["particles"] = ("particles",)
OUTPUT_TYPES["Plume3D"] = "volume"
INPUT_TYPES.update({f"geo{i}": ("geometry",) for i in range(8)})
INPUT_TYPES.update({f"light{i}": ("light",) for i in range(8)})
LIMITS = {"splat_write_overwrite": (0, 1), "flip_winding": (0, 1), "recompute_normals": (0, 1),
          "displace_scale": (-1000000.0, 1000000.0), "displace_offset": (-1000000.0, 1000000.0),
          "splat_relight": (0.0, 1.0), "splat_shadow_catch": (0.0, 1.0), "splat_specular": (0.0, 1.0), "splat_normal_smoothing": (0, 64), "splat_delight_iterations": (1, 200), "splat_delight_smoothness": (0.0, 1.0), "splat_delight_light_order": (0, 2), "splat_sh_degree": (0, 3), "splat_opacity": (0.0, 1000000.0),
          "splat_scale": (0.000001, 1000000.0), "uscale": (0.000001, 1000000.0),
          "pivot_x": (-1000000.0, 1000000.0), "pivot_y": (-1000000.0, 1000000.0),
          "pivot_z": (-1000000.0, 1000000.0), "width": (1, 8192), "height": (1, 8192), "size": (1, 4096),
          "exposure": (-20, 20), "multiply": (-100, 100), "offset": (-100, 100),
          "red": (-100, 100), "green": (-100, 100), "blue": (-100, 100),
          "alpha": (0, 1), "mix": (0, 1),
          "x": (-8192, 8192), "y": (-8192, 8192), "subimage": (0, 1023),
          "translate_x": (-8192.0, 8192.0), "translate_y": (-8192.0, 8192.0),
          "rotate": (-100000.0, 100000.0), "scale": (0.001, 1000.0),
          "center_x": (-8192.0, 8192.0), "center_y": (-8192.0, 8192.0),
          "lift": (-10, 10), "gamma": (0.01, 100), "gain": (0, 100), "saturation": (0, 10),
          "radius": (0, 500), "which": (0, 1),
          "invert": (0, 1), "reference_frame": (-1000000, 1000000),
          "apply_translate": (0, 1), "apply_rotate": (0, 1), "apply_scale": (0, 1),
          "frame_offset": (-1000000, 1000000),
          "minimum": (-1000000.0, 1000000.0), "maximum": (-1000000.0, 1000000.0),
          "clamp_min": (0, 1), "clamp_max": (0, 1), "invert_mask": (0, 1),
          # Erode/Dilate: signed, matching Nuke's own Erode (fast) "size" range.
          "erode_size": (-1000.0, 1000.0), "dilate_size": (-1000.0, 1000.0),
          "median_size": (0.0, 500.0), "sharpen_amount": (0.0, 10.0), "sharpen_size": (0.0, 500.0),
          "glow_threshold": (-10.0, 10.0), "glow_size": (0.0, 500.0), "soften_size": (0.0, 500.0), "defocus": (0.0, 500.0), "aspect": (0.1, 10.0), "angle": (-360.0, 360.0), "length": (0.0, 1000.0), "distance": (0.0, 2000.0), "shadow_size": (0.0, 500.0), "opacity": (0.0, 1.0), "numpixels": (-8192, 8192), "clip_to_format": (0, 1), "blackpoint": (-100.0, 100.0), "gang": (0, 1), "brightness": (0.0, 100.0),
          "edgeblur_size": (0.0, 500.0), "edge_mult": (0.0, 10.0), "extend_size": (0.0, 500.0), "extend_threshold": (0.0, 1.0),
          "wrap_diffuse": (0.0, 500.0), "fgblur": (0.0, 500.0), "bgblur": (0.0, 500.0), "wrap_threshold": (-10.0, 10.0), "use_constant_highlight": (0, 1),
          "bits": (1, 16), "dither_amount": (0.0, 4.0),
          "red_size": (0.0, 500.0), "green_size": (0.0, 500.0), "blue_size": (0.0, 500.0),
          "red_intensity": (0.0, 10.0), "green_intensity": (0.0, 10.0), "blue_intensity": (0.0, 10.0),
          "luminance_weighted": (0, 1), "black": (0.0, 1.0), "colors": (2, 65536),
          "softclip_min": (-10.0, 10.0), "softclip_max": (0.0, 1000.0),
          "hue_range_min": (0.0, 360.0), "hue_range_max": (0.0, 360.0), "hue_rolloff": (0.0, 360.0),
          "hue_rotation": (-360.0, 360.0), "saturation_range_min": (0.0, 1.0), "saturation_range_max": (0.0, 1.0),
          "saturation_rolloff": (0.0, 1.0), "sat_adjust": (-1.0, 10.0), "set_saturation": (0, 1),
          "brightness_range_min": (0.0, 1000.0), "brightness_range_max": (0.0, 1000.0), "brightness_rolloff": (0.0, 1000.0),
          "brt_adjust": (-1.0, 100.0), "set_brightness": (0, 1), "output_alpha": (0, 1), "normalize": (0, 1),
          "weight0": (-100.0, 100.0), "weight1": (-100.0, 100.0), "weight2": (-100.0, 100.0), "weight3": (-100.0, 100.0),
          "weight4": (-100.0, 100.0), "weight5": (-100.0, 100.0), "weight6": (-100.0, 100.0), "weight7": (-100.0, 100.0),
          "area_x": (-16384.0, 16384.0), "area_y": (-16384.0, 16384.0), "area_r": (-16384.0, 16384.0), "area_t": (-16384.0, 16384.0),
          "uv_scale_x": (-1000.0, 1000.0), "uv_scale_y": (-1000.0, 1000.0),
          "uv_offset_x": (-10000.0, 10000.0), "uv_offset_y": (-10000.0, 10000.0),
          "bundle": (0, 1),
          "vector_scale": (-100.0, 100.0), "vector_offset": (-10.0, 10.0), "max_length": (0.0, 1000.0),
          "flip_x": (0, 1), "flip_y": (0, 1),
          # Ramp/Radial/Rectangle/Noise/Text (group c2 Draw generators).
          "p0_x": (-8192.0, 8192.0), "p0_y": (-8192.0, 8192.0),
          "p1_x": (-8192.0, 8192.0), "p1_y": (-8192.0, 8192.0),
          "color0_red": (-100.0, 100.0), "color0_green": (-100.0, 100.0), "color0_blue": (-100.0, 100.0),
          "color0_alpha": (0.0, 1.0),
          "color1_red": (-100.0, 100.0), "color1_green": (-100.0, 100.0), "color1_blue": (-100.0, 100.0),
          "color1_alpha": (0.0, 1.0),
          "box_x": (-8192.0, 8192.0), "box_y": (-8192.0, 8192.0),
          "box_width": (0.0, 16384.0), "box_height": (0.0, 16384.0),
          # Grid: spacing in pixels, line counts, offset and line width.
          "spacing_x": (1.0, 16384.0), "spacing_y": (1.0, 16384.0), "number_x": (0, 4096), "number_y": (0, 4096),
          "grid_offset_x": (-16384.0, 16384.0), "grid_offset_y": (-16384.0, 16384.0), "line_width": (0.0, 4096.0),
          # Shared fraction-of-the-box-radius softness for both Radial and Rectangle.
          "softness": (0.0, 1.0),
          "z_slice": (-100000.0, 100000.0), "octaves": (1, 8), "lacunarity": (0.01, 8.0),
          "seed": (0, 2147483647), "font_size": (1.0, 2000.0),
          # Keyer's four-point range ramp (group c3): unbounded like Clamp's minimum/maximum,
          # since a keyed quantity can legally sit outside 0..1 on HDR footage.
          "range_a": (-1000000.0, 1000000.0), "range_b": (-1000000.0, 1000000.0),
          "range_c": (-1000000.0, 1000000.0), "range_d": (-1000000.0, 1000000.0),
          # HueCorrect (step 4a): per-band multipliers and a hue rotation in degrees; ColorMatrix's
          # nine matrix entries are unbounded in practice, so the range is only a sanity fence.
          "sat_red": (0.0, 10.0), "lum_red": (0.0, 10.0), "sat_yellow": (0.0, 10.0), "lum_yellow": (0.0, 10.0), "sat_green": (0.0, 10.0), "lum_green": (0.0, 10.0), "sat_cyan": (0.0, 10.0), "lum_cyan": (0.0, 10.0), "sat_blue": (0.0, 10.0), "lum_blue": (0.0, 10.0), "sat_magenta": (0.0, 10.0), "lum_magenta": (0.0, 10.0),
          "hue_shift": (-360.0, 360.0),
          "matrix_00": (-1000.0, 1000.0), "matrix_01": (-1000.0, 1000.0), "matrix_02": (-1000.0, 1000.0), "matrix_10": (-1000.0, 1000.0), "matrix_11": (-1000.0, 1000.0), "matrix_12": (-1000.0, 1000.0), "matrix_20": (-1000.0, 1000.0), "matrix_21": (-1000.0, 1000.0), "matrix_22": (-1000.0, 1000.0),
          # HueKeyer's simplified hue + saturation range.
          "hue_center": (0.0, 360.0), "hue_width": (0.0, 360.0), "hue_softness": (0.0, 180.0),
          "sat_min": (0.0, 1.0), "sat_max": (0.0, 1.0),
          # TimeOffset/FrameHold/Retime (group c4): frame counts share `frame_offset`'s wide
          # integer range; "increment" is non-negative (0 means "no progression", Nuke's own
          # FrameHold default); "reverse" is a plain 0/1 flag like "invert".
          "time_offset": (-1000000, 1000000), "reverse": (0, 1),
          "first_frame": (-1000000, 1000000), "increment": (0, 1000000),
          "input_range_start": (-1000000, 1000000), "input_range_end": (-1000000, 1000000),
          "output_range_start": (-1000000, 1000000), "output_range_end": (-1000000, 1000000),
          "speed": (-1000.0, 1000.0),
          # TimeClip/FrameRange/AppendClip (step 4b): frame counts again; "dissolve" and the
          # per-clip lengths are non-negative frame counts (length 0 skips that clip).
          "first": (-1000000, 1000000), "last": (-1000000, 1000000), "last_frame": (-1000000, 1000000),
          "dissolve": (0, 1000000), **{f"length{i}": (0, 1000000) for i in range(8)},
          # Reformat (group 2c5): pixel_aspect is a ratio close to 1; center/flip/flop/turn/
          # preserve_bbox are the codebase's usual 0/1 flags.
          "pixel_aspect": (0.01, 100.0), "center": (0, 1), "flip": (0, 1), "flop": (0, 1),
          "turn": (0, 1), "preserve_bbox": (0, 1),
          # CornerPin (group 2c5): eight point pairs share Transform's translate_x/translate_y
          # range, since they are the same kind of quantity -- a pixel position in the canvas.
          "from1_x": (-8192.0, 8192.0), "from1_y": (-8192.0, 8192.0),
          "from2_x": (-8192.0, 8192.0), "from2_y": (-8192.0, 8192.0),
          "from3_x": (-8192.0, 8192.0), "from3_y": (-8192.0, 8192.0),
          "from4_x": (-8192.0, 8192.0), "from4_y": (-8192.0, 8192.0),
          "to1_x": (-8192.0, 8192.0), "to1_y": (-8192.0, 8192.0),
          "to2_x": (-8192.0, 8192.0), "to2_y": (-8192.0, 8192.0),
          "to3_x": (-8192.0, 8192.0), "to3_y": (-8192.0, 8192.0),
          "to4_x": (-8192.0, 8192.0), "to4_y": (-8192.0, 8192.0)}
LIMITS.update({"diffuse": (0.0, 1.0), "specular": (0.0, 1.0)})
LIMITS.update({"plume_resolution": (4, 128), "plume_seed": (0, 2147483647)})
LIMITS.update({"volume_step_size": (0.0005, 100.0), "volume_density_scale": (0.0, 100000.0),
               "volume_shadow_density": (0.0, 100000.0), "volume_shadow_steps": (1, 256),
               "volume_scattering": (0.0, 1000.0), "volume_absorption": (0.0, 1000.0),
               "volume_red": (0.0, 1000.0), "volume_green": (0.0, 1000.0), "volume_blue": (0.0, 1000.0),
               "volume_fps": (0.001, 1000.0), "volume_depth_threshold": (0.0, 100000.0)})
# Particle knobs (ParticleEmitter3D, ParticleCache3D). Variances are fractions: a value of 0.25 spreads
# the knob by plus or minus 25 percent. start_frame may be negative for pre-roll.
LIMITS.update({"emit_rate": (0.0, 10000000.0), "start_frame": (-1000000, 1000000),
               "life": (0.0, 1000000.0), "life_variance": (0.0, 1.0),
               "emit_speed": (-1000000.0, 1000000.0), "speed_variance": (0.0, 1.0),
               "emit_dir_x": (-1000000.0, 1000000.0), "emit_dir_y": (-1000000.0, 1000000.0),
               "emit_dir_z": (-1000000.0, 1000000.0), "direction_from_normals": (0, 1),
               "spread": (0.0, 180.0), "particle_size": (0.0, 1000000.0), "size_variance": (0.0, 1.0),
               "substeps": (1, 64), "max_particles": (1, 10000000),
               "cache_memory_mb": (1, 1048576), "cache_disk_mb": (0, 10485760)})
# Particle forces (step 2b): accelerations are units per frame squared.
LIMITS.update({"probability": (0.0, 1.0), "from_frame": (-1000000, 1000000), "to_frame": (-1000000, 1000000),
               "strength": (-1000000.0, 1000000.0), "drag": (0.0, 1000000.0),
               "drag_quadratic": (0.0, 1000000.0), "wind_gust": (0.0, 1.0),
               "wind_gust_rate": (0.0, 1000.0), "turb_size": (0.0001, 1000000.0),
               **{name: (-1000000.0, 1000000.0) for name in
                  ("gravity_x", "gravity_y", "gravity_z", "wind_x", "wind_y", "wind_z")}})
# Bounce and rendering (step 2c). `friction` is a Coulomb coefficient, so it may exceed 1.
LIMITS.update({"bounce": (0.0, 2.0), "friction": (0.0, 100.0), "kill_on_collision": (0, 1),
               "size_scale": (0.0, 1000000.0)})
LIMITS.update({name: (-1000000.0, 1000000.0) for name in
               ("tx", "ty", "tz", "rx", "ry", "rz", "roll", "target_x", "target_y", "target_z")})
LIMITS.update({"sx": (0.001, 1000.0), "sy": (0.001, 1000.0), "sz": (0.001, 1000.0),
               "card_width": (0.001, 100000.0), "card_height": (0.001, 100000.0),
               "cube_size": (0.001, 100000.0), "sphere_radius": (0.001, 100000.0),
               "segments": (3, 128),
               # Nuke's own knob names, shared by Card3D, Sphere3D and Cylinder3D. Card3D's
               # default is 1x1 (today's single quad), so the shared range allows 1; Sphere3D
               # and Cylinder3D need at least 2 rows / 3 columns for a closed shape, which the
               # geometry code itself floors, the same way "segments" always did.
               "rows": (1, 128), "columns": (1, 128),
               "cyl_radius": (0.001, 100000.0), "cyl_height": (0.001, 100000.0),
               "intensity": (0.0, 1000.0),
               "cone_angle": (1.0, 180.0), "cone_penumbra_angle": (0.0, 90.0),
               "cone_falloff": (0.0, 10.0),
               "shadow_bias": (0.0, 1.0), "shadow_blur": (0.0, 45.0), "shadow_samples": (1, 64),
               "ambient": (0.0, 10.0),
               "spec_amount": (0.0, 1.0), "spec_shininess": (1.0, 1024.0),
               "emission": (0.0, 1000.0), "samples": (1, 4),
               "focal": (0.01, 100000.0), "haperture": (0.01, 100000.0),
               "vaperture": (0.01, 100000.0), "near": (0.0001, 1000000.0), "far": (0.001, 1000000.0)})

# Declared artifact type per node kind. The cache does not yet *store* the type, so this is the
# declaration the scheduler reads, not a claim that typed storage exists.
ARTIFACT_TYPES = {"Roto": "matte"}
DEFAULT_ARTIFACT_TYPE = "image"


def artifact_type(kind):
    if kind not in SPECS:
        raise ValueError(f"Unknown node type: {kind}")
    return ARTIFACT_TYPES.get(kind, DEFAULT_ARTIFACT_TYPE)

# Document time range limits (schema v5). FPS is stored from v5 onward so the field exists for a
# future clip timeline; nothing consumes it yet — see docs/TIME_MODEL.md.
TIME_LIMITS = {"first": (-1000000, 1000000), "last": (-1000000, 1000000),
               "current": (-1000000, 1000000), "fps": (0.01, 1000.0)}
DEFAULT_TIME = {"first": 1, "last": 1, "current": 1, "fps": 24.0}
MISSING_FRAME_POLICIES = ("error", "hold", "black")


# Merge keeps A as foreground and B as background, mirroring Nuke's wiring convention.
MERGE_OPERATIONS = ("over", "under", "plus", "minus", "multiply", "screen", "max", "min",
                    "difference", "divide", "mask", "stencil", "in", "out", "atop", "xor",
                    "average", "from", "hypot", "matte", "disjoint-over", "conjoint-over", "copy",
                    "exclusion", "geometric", "overlay", "hard-light", "soft-light", "color-dodge",
                    "color-burn")
TRANSFORM_FILTERS = ("nearest", "bilinear", "cubic")
TRACKER_MODES = ("match_move", "stabilise")
# Write destinations. Deliberately the two formats media.py can actually write; a format list
# longer than the writer is a promise the render button cannot keep.
WRITE_FILE_TYPES = ("Auto", "exr", "png")
EXR_BIT_DEPTHS = ("half", "float")
# Each ChannelShuffle output names its source explicitly. "0"/"1" are constants; there is no
# "leave it alone" option, because that is the one that hides a mistake.
CHANNEL_SOURCES = ("A.r", "A.g", "A.b", "A.a", "B.r", "B.g", "B.b", "B.a", "0", "1")
CHOICES = {"before": ["hold", "loop", "bounce", "black"], "after": ["hold", "loop", "bounce", "black"],
           "frame_range_type": ["custom", "all"],
           "splat_orientation": ["as_authored", "colmap"],
           "splat_colorspace": ["srgb", "linear"],
           "rot_order": ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"], "colorspace": ["Auto", "sRGB", "Linear Rec.709", "ACEScg", "ACES2065-1", "Raw"],
           "alpha_mode": ["Auto", "Straight", "Premultiplied"],
           "operation": list(MERGE_OPERATIONS),
           "filter": list(TRANSFORM_FILTERS),
           "red_from": ["R", "G", "B", "A", "0", "1"], "green_from": ["R", "G", "B", "A", "0", "1"],
           "blue_from": ["R", "G", "B", "A", "0", "1"], "alpha_from": ["R", "G", "B", "A", "0", "1"],
           "missing": list(MISSING_FRAME_POLICIES),
           "justify": ["left", "center", "right"],
           "blur_type": ["linear", "radial", "zoom"],
           "highlight_merge": ["plus", "screen", "max", "over"],
           "conversion": ["none", "preserve hue and brightness", "preserve hue and saturation", "logarithmic compress"],
           "u_channel": ["R", "G", "B", "A"], "v_channel": ["R", "G", "B", "A"],
           "uv_outside": ["black", "clamp"], "vector_method": ["forward", "backward"],
           "vector_alpha": ["none", "weighted"],
           "mode": list(TRACKER_MODES), "exposure_mode": ["stops", "densities"],
           "out_red": list(CHANNEL_SOURCES), "out_green": list(CHANNEL_SOURCES),
           "out_blue": list(CHANNEL_SOURCES), "out_alpha": list(CHANNEL_SOURCES),
           # Write output format. "Auto" reads the extension on the path rather than second-guessing
           # it, so renaming output.exr to output.png changes the writer and nothing else.
           "file_type": list(WRITE_FILE_TYPES), "bit_depth": list(EXR_BIT_DEPTHS),
           "project_outside": ["transparent", "clamp"], "project_backfaces": ["project", "skip"],
           "project_occlusion": ["off", "depth"],
           "shadows": ["off", "on"], "splat_cast_shadows": ["on", "off"], "splat_delight": ["off", "on"], "splat_use_intrinsics": ["on", "off"], "use_intrinsics": ["on", "off"],
           "cyl_caps": ["closed", "open"],
           "emit_from": ["point", "vertices", "surface", "volume"],
           "emit_rate_unit": ["per_frame", "per_second"],
           "turb_mode": ["curl", "gradient"],
           "representation": ["points", "spheres", "cards"],
           "normals_mode": ["unchanged", "recompute", "flip", "unify"],
           "displace_channel": ["luminance", "red", "green", "blue", "alpha"],
           "render_backend": ["cpu", "auto", "gpu"],
           "render_mode": ["raster", "raytrace"],
           "light_type": ["Directional", "Point", "Spot"],
           "falloff_type": ["No falloff", "Linear", "Quadratic", "Cubic"], "render_output": ["rgba", "depth", "normals", "albedo", "diffuse",
                             "specular", "emission", "position", "uv", "object_id", "relight", "splats", "normals_blend",
                             "multichannel"],
           # Invert/Clamp/Multiply/Add/Gamma's channel selector. "rgba" also inverts/clamps alpha.
           "channels": ["rgb", "rgba", "alpha"],
           # Copy: which of A's channels replaces each of B's; "none" leaves that channel as B's own.
           "copy_red": ["none", "A.r", "A.g", "A.b", "A.a"], "copy_green": ["none", "A.r", "A.g", "A.b", "A.a"],
           "copy_blue": ["none", "A.r", "A.g", "A.b", "A.a"], "copy_alpha": ["none", "A.r", "A.g", "A.b", "A.a"],
           # ChannelMerge: single-channel source/destination selectors.
           "a_channel": ["A.r", "A.g", "A.b", "A.a"], "b_channel": ["B.r", "B.g", "B.b", "B.a"],
           "out_channel": ["R", "G", "B", "A"],
           # Keyer's keyed quantity (group c3).
           "keyer_operation": ["luminance", "red", "green", "blue", "saturation", "min", "max"],
           # Reformat (group 2c5).
           "reformat_type": ["to_format", "scale", "to_box"],
           "format": list(REFORMAT_FORMATS) + ["Custom"],
           "resize_type": ["none", "width", "height", "fit", "fill", "distort"],
           # CornerPin (group 2c5).
           "direction": ["forward", "inverse"]}
CHOICES["volumes"] = ["on", "off"]
# Before "multichannel", which stays the menu's last entry (a graph-level output, not a render() one).
_MULTICHANNEL = CHOICES["render_output"].index("multichannel")
CHOICES["render_output"][_MULTICHANNEL:_MULTICHANNEL] = ["volume_density", "volume_motion", "volume_temperature", "volume_vorticity"]


def _downstream_of(nodes, key):
    """Every node reachable by following outputs from `key`. Iterative, so a long chain cannot
    blow the recursion limit, and used to keep a Viewer rewire from closing a cycle."""
    reached, pending = set(), [key]
    while pending:
        current = pending.pop()
        for other_key, other in nodes.items():
            if other_key in reached:
                continue
            if current in other["inputs"].values():
                reached.add(other_key)
                pending.append(other_key)
    return reached


def viewer_state(doc):
    """The viewer's inputs and compare choice, normalised: a document without the optional keys
    reads as input 1 = the viewed node, A only."""
    viewer = doc["settings"]["viewer"]
    if "inputs" in viewer:
        return {"inputs": list(viewer["inputs"]), "active": viewer["active"], "b": viewer["b"],
                "compare": viewer["compare"]}
    return {"inputs": [doc["view"]] + [None] * (VIEWER_INPUT_COUNT - 1), "active": 1, "b": None,
            "compare": COMPARE_MODES[0]}


def viewer_look(doc):
    """The viewer's display-only adjustments, normalised: a document without `look` reads as the
    default (no gain, gamma 1, zebra off, the project's view)."""
    return {**VIEWER_LOOK_DEFAULT, **doc["settings"]["viewer"].get("look", {})}


def viewer_roi(doc):
    roi = doc["settings"]["viewer"].get("roi", VIEWER_ROI_DEFAULT)
    return {"on": roi["on"], "rect": list(roi["rect"])}


def viewer_proxy(doc):
    return doc["settings"]["viewer"].get("proxy", 1)


def viewer_masks(doc):
    return {**VIEWER_MASKS_DEFAULT, **doc["settings"]["viewer"].get("masks", {})}


def _validate_roi(roi):
    if not isinstance(roi, dict) or set(roi) != {"on", "rect"} or type(roi["on"]) is not bool:
        raise ValueError("settings.viewer.roi must be {on: boolean, rect: [x0, y0, x1, y1]}")
    rect = roi["rect"]
    if (not isinstance(rect, list) or len(rect) != 4
            or any(type(value) not in (int, float) or not 0.0 <= value <= 1.0 for value in rect)
            or not rect[0] < rect[2] or not rect[1] < rect[3]):
        raise ValueError("settings.viewer.roi.rect must be [x0, y0, x1, y1] fractions, 0 to 1, with x0 < x1 and y0 < y1")


def _validate_masks(masks):
    if not isinstance(masks, dict) or set(masks) != set(VIEWER_MASKS_DEFAULT):
        raise ValueError(f"settings.viewer.masks must define {sorted(VIEWER_MASKS_DEFAULT)}")
    if masks["mask"] not in VIEWER_MASKS:
        raise ValueError(f"settings.viewer.masks.mask must be one of {list(VIEWER_MASKS)}")
    if masks["mode"] not in VIEWER_MASK_MODES:
        raise ValueError(f"settings.viewer.masks.mode must be one of {list(VIEWER_MASK_MODES)}")


def _validate_look(look):
    if not isinstance(look, dict) or set(look) != set(VIEWER_LOOK_DEFAULT):
        raise ValueError(f"settings.viewer.look must define {sorted(VIEWER_LOOK_DEFAULT)}")
    for name, (low, high) in (("gain", VIEWER_GAIN_RANGE), ("gamma", VIEWER_GAMMA_RANGE)):
        value = look[name]
        if type(value) not in (int, float) or not low <= value <= high:
            raise ValueError(f"settings.viewer.look.{name} must be a number from {low} to {high}")
    if type(look["zebra"]) is not bool:
        raise ValueError("settings.viewer.look.zebra must be boolean")
    if type(look["display"]) is not str or not look["display"]:
        raise ValueError("settings.viewer.look.display must be a view name")


def _store_viewer_state(doc, state):
    """Write the state canonically: the default is stored as the absence of the keys."""
    viewer = doc["settings"]["viewer"]
    default = {"inputs": [doc["view"]] + [None] * (VIEWER_INPUT_COUNT - 1), "active": 1, "b": None,
               "compare": COMPARE_MODES[0]}
    if state == default:
        for name in VIEWER_STATE_KEYS:
            viewer.pop(name, None)
    else:
        viewer.update(inputs=list(state["inputs"]), active=state["active"], b=state["b"],
                      compare=state["compare"])


def _apply_view(doc, target):
    nodes = doc["nodes"]
    doc["view"] = target
    # Nuke parity: a Viewer node's input *is* whatever is being viewed, so viewing a node
    # rewires every Viewer to it instead of leaving a stale connection that claims the
    # comp is wired somewhere it is not. Viewing nothing disconnects them, for the same
    # reason. A Viewer is skipped when the target sits downstream of it, because that
    # connection would be a cycle and `validate` would reject the whole edit.
    for viewer_key, viewer in nodes.items():
        if viewer["type"] != "Viewer":
            continue
        if target is not None and (target == viewer_key
                                   or target in _downstream_of(nodes, viewer_key)):
            continue
        viewer["inputs"]["image"] = target


def _viewer_slot(value, name="slot"):
    if type(value) is not int or not 1 <= value <= VIEWER_INPUT_COUNT:
        raise ValueError(f"{name} must be an integer from 1 to {VIEWER_INPUT_COUNT}")
    return value


def upgrade_document(document):
    doc = copy.deepcopy(document)
    if isinstance(doc, dict) and doc.get("version") == 1:
        for node in doc.get("nodes", {}).values():
            if node.get("type") == "Read":
                node["params"] = {**SPECS["Read"]["params"], **node["params"]}
        doc["version"] = 2
    if isinstance(doc, dict) and doc.get("version") == 2:
        # v2 -> v3: rename Transform x/y to translate_x/translate_y (int -> float) and add the
        # rotate/scale/center/filter params with identity defaults; add Merge.operation = "over"
        # so v0.3.0 comps render identically and existing projects opt into the new ops explicitly.
        for node in doc.get("nodes", {}).values():
            kind = node.get("type")
            if kind == "Transform":
                old_params = node.get("params", {})
                upgraded = {
                    "translate_x": float(old_params.get("x", 0)),
                    "translate_y": float(old_params.get("y", 0)),
                    "rotate": 0.0,
                    "scale": 1.0,
                    "center_x": 0.0,
                    "center_y": 0.0,
                    "filter": "nearest",
                }
                upgraded.update({k: v for k, v in old_params.items() if k not in ("x", "y")})
                node["params"] = upgraded
            elif kind == "Merge":
                params = node.get("params", {})
                if "operation" not in params:
                    params = {**params, "operation": "over"}
                    node["params"] = params
        doc["version"] = 3
    if isinstance(doc, dict) and doc.get("version") == 3:
        # v3 -> v4: every image-filter node (Grade, ColorCorrect, Blur, Transform, Crop) gains an
        # optional "mask" input slot and a per-node "mix" param (defaults to 1.0 so existing
        # projects render byte-identically). Dot and Switch are new in v4; no upgrade path
        # synthesizes them — they appear only when the user adds them.
        for node in doc.get("nodes", {}).values():
            kind = node.get("type")
            if kind in IMAGE_FILTER_KINDS:
                inputs = node.setdefault("inputs", {})
                if "mask" not in inputs:
                    inputs["mask"] = None
                params = node.setdefault("params", {})
                if "mix" not in params:
                    params["mix"] = 1.0
        doc["version"] = 4
    if isinstance(doc, dict) and doc.get("version") == 4:
        # v4 -> v5: the document gains a time range, and Read gains its source-time mapping.
        # A v4 comp had no time axis at all, so it upgrades to a single-frame range — every
        # still-image Read renders byte-identically at frame 1, which is the only frame there is.
        # frame_offset=0 and missing="error" preserve v4 behaviour for literal (non-pattern) paths.
        doc["time"] = dict(DEFAULT_TIME)
        for node in doc.get("nodes", {}).values():
            if node.get("type") == "Read":
                params = node.setdefault("params", {})
                params.setdefault("frame_offset", 0)
                params.setdefault("missing", "error")
        doc["version"] = 5
    if isinstance(doc, dict) and doc.get("version") == 5:
        # v5 -> v6: the document gains a top-level "animation" section. v5 had no curves, so the
        # upgrade is just an empty {"curves": {}} shape. Animation never changes a stored
        # node["params"]; it only adds an evaluation-time override layer, so v5 graphs render
        # byte-identically after upgrade.
        doc["animation"] = {"curves": {}}
        doc["version"] = 6
    if isinstance(doc, dict) and doc.get("version") == 6:
        # v6 -> v7: make the colour contract explicit in the project. The processing space is
        # ACEScg and the default view is the ACES 2.0 SDR Rec.709 transform; the viewer background
        # also becomes a saved project preference. Existing comps previously used an implicit
        # sRGB view, so upgrading them keeps that view to preserve their appearance.
        doc["settings"] = copy.deepcopy(DEFAULT_SETTINGS)
        doc["settings"]["color"]["view"] = "sRGB"
        # The literal 7, never SCHEMA_VERSION: a step must write the version it actually emits.
        # If this said SCHEMA_VERSION, the day a v8 lands this block would stamp a document "8"
        # while having done only v7's work, and the v7 -> v8 step below it would never fire — a
        # document tagged with the new version but missing the new section. Every step above
        # writes its own literal for the same reason.
        doc["version"] = 7
    if isinstance(doc, dict) and doc.get("version") == 7:
        # v7 -> v8: the document gains `node_data`, a generic per-node structured payload keyed by
        # node id the way animation.curves is. No v7 node type carries a payload, so every existing
        # comp upgrades to an empty section and renders byte-identically.
        doc["node_data"] = {}
        # The literal 8, for the reason spelled out on the v6 -> v7 step above.
        doc["version"] = 8
    if isinstance(doc, dict) and doc.get("version") == 8:
        # v8 -> v9: the document gains `expressions`, the third and last way a parameter gets a
        # value (stored number, v6 curve, formula). No v8 document has any, so every existing comp
        # upgrades to an empty section and renders byte-identically.
        doc["expressions"] = {}
        # The literal 9, for the reason spelled out on the v6 -> v7 step above.
        doc["version"] = 9
    if isinstance(doc, dict) and doc.get("version") == 9:
        # v9 -> v10: ordered agent reference tags. No v9 document carries tags, so the empty
        # list preserves both graph evaluation and the serialized meaning of every node.
        doc["references"] = []
        doc["version"] = 10
    if isinstance(doc, dict) and doc.get("version") == 10:
        # v10 -> v11: Merge gains an optional "mask" input. An unwired mask is full opacity, so
        # every existing comp renders byte-identically.
        for node in doc.get("nodes", {}).values():
            if node.get("type") == "Merge":
                node.setdefault("inputs", {}).setdefault("mask", None)
        doc["version"] = 11
    if isinstance(doc, dict) and doc.get("version") == 11:
        # v11 -> v12: nodes may carry optional `label` and `thumbnail` fields. Absent means the
        # default, so every v11 node is already a valid v12 node and renders byte-identically.
        doc["version"] = 12
    # Additive 3D options preserve existing rendering behavior.
    if isinstance(doc, dict) and doc.get("version") == SCHEMA_VERSION:
        # The document-wide format registry (lane L2 step 4c) is additive like the options below:
        # an old document gets the built-in list Reformat always had, so no window changes on load.
        settings = doc.get("settings")
        if isinstance(settings, dict):
            settings.setdefault("formats", builtin_formats())
        nodes = doc.get("nodes", {})
        if isinstance(nodes, dict):
            for node in nodes.values():
                if isinstance(node, dict) and node.get("type") in ("Card3D", "Cube3D", "Sphere3D", "ReadGeo3D"):
                    params = node.get("params")
                    if isinstance(params, dict):
                        for key in ("spec_amount", "spec_shininess", "emission"):
                            params.setdefault(key, _SURFACE[key])
                # Uniform scale, rotation order and pivot default to the identity, so a node saved
                # before they existed keeps its matrix exactly.
                if isinstance(node, dict) and node.get("type") in (*GEOMETRY_TYPES, "Scene3D"):
                    params = node.get("params")
                    if isinstance(params, dict):
                        for key in _XFORM_ADDED:
                            params.setdefault(key, _XFORM[key])
                # Card3D gains rows/columns (lane L3 step 2): 1x1 is the single quad every old
                # document already rendered, so this is byte-identical.
                if isinstance(node, dict) and node.get("type") == "Card3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("rows", 1)
                        params.setdefault("columns", 1)
                # Sphere3D gains rows/columns, replacing the single "segments" knob with Nuke's
                # own pair. Derived from the node's own stored "segments" (not the SPECS
                # default), the same split `_sphere` always used, so an old document with a
                # non-default segments value keeps its resolution exactly.
                if isinstance(node, dict) and node.get("type") == "Sphere3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        segments = params.get("segments", 32)
                        params.setdefault("columns", max(3, int(segments)))
                        params.setdefault("rows", max(2, int(segments) // 2))
                if isinstance(node, dict) and node.get("type") == "Render3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("render_backend", "cpu")
                        params.setdefault("render_mode", "raster")
                        params.setdefault("passes", "beauty,normals,depth")
                        for name, default in _VOLUME_RENDER_DEFAULTS.items():
                            params.setdefault(name, default)
                if isinstance(node, dict) and node.get("type") == "ReadSplat3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("splat_relight", 0.0)
                        params.setdefault("splat_shadow_catch", 0.0)
                        params.setdefault("splat_cast_shadows", "on")
                        params.setdefault("splat_specular", 0.0)
                        params.setdefault("splat_normal_smoothing", 0)
                        params.setdefault("splat_delight", "off")
                        params.setdefault("splat_delight_iterations", 12)
                        params.setdefault("splat_delight_smoothness", 0.5)
                        params.setdefault("splat_delight_light_order", 1)
                        params.setdefault("splat_use_intrinsics", "on")
                if isinstance(node, dict) and node.get("type") == "Light3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("shadows", "off")
                        # Spot cone and falloff (lane L3 step 3): a Directional or Point light
                        # ignores the cone, and "No falloff" is what lights always did.
                        params.setdefault("cone_angle", 30.0)
                        params.setdefault("cone_penumbra_angle", 5.0)
                        params.setdefault("cone_falloff", 1.0)
                        params.setdefault("falloff_type", "No falloff")
                        # Shadow offset and blur (lane L4 step B): today's epsilon, a hard shadow.
                        params.setdefault("shadow_bias", 0.001)
                        params.setdefault("shadow_blur", 0.0)
                        params.setdefault("shadow_samples", 1)
                if isinstance(node, dict) and node.get("type") == "Camera3D":
                    _camera_fov_to_film_back(doc, node)
                if isinstance(node, dict) and node.get("type") == "Project3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("project_occlusion", "off")
                # Step 5c: Shuffle's layer (empty = the input's own channels) and Write's bundle
                # (off = no manifest) default to what every old node already did.
                for kind, key, default in (("Shuffle", "layer", ""), ("Write", "bundle", 0)):
                    if isinstance(node, dict) and node.get("type") == kind:
                        params = node.get("params")
                        if isinstance(params, dict):
                            params.setdefault(key, default)
    return doc


def _camera_fov_to_film_back(doc, node):
    """Camera3D gained a film back (lane L3 step 3) and lost its stored `fov`. An old node keeps
    its exact vertical field of view: the film back defaults to Nuke's aperture and the focal
    length is solved from the stored `fov`. Keys of an animated `fov` curve become focal-length
    keys the same way (exact at the keys and for constant curves; linear curves interpolate the
    focal length, not the angle, between keys)."""
    params = node.get("params")
    if not isinstance(params, dict):
        return
    params.setdefault("haperture", filmback.DEFAULT_HAPERTURE)
    params.setdefault("vaperture", filmback.DEFAULT_VAPERTURE)
    vaperture = params["vaperture"]
    if "fov" in params:
        fov = params.pop("fov")
        if "focal" not in params:
            params["focal"] = filmback.focal_from_fov(fov, vaperture)
    params.setdefault("focal", filmback.DEFAULT_FOCAL)
    curves = doc.get("animation", {}).get("curves", {}) if isinstance(doc.get("animation"), dict) else {}
    for node_id, node_curves in curves.items():
        if doc["nodes"].get(node_id) is node and isinstance(node_curves, dict) and "fov" in node_curves:
            curve = node_curves.pop("fov")
            for key in curve.get("keys", ()):
                key["value"] = filmback.focal_from_fov(key["value"], vaperture)
            node_curves["focal"] = curve
    expressions = doc.get("expressions")
    if isinstance(expressions, dict):
        # A formula written in degrees cannot be rewritten as millimetres; the static focal
        # length above still holds the node's stored field of view.
        for node_id, node_exprs in list(expressions.items()):
            if doc["nodes"].get(node_id) is node and isinstance(node_exprs, dict):
                node_exprs.pop("fov", None)
                if not node_exprs:
                    del expressions[node_id]


def empty_document():
    return {"version": SCHEMA_VERSION, "nodes": {}, "view": None, "time": dict(DEFAULT_TIME),
            "animation": {"curves": {}},
            "settings": {**copy.deepcopy(DEFAULT_SETTINGS), "formats": builtin_formats()},
            "node_data": {}, "expressions": {}, "references": []}


def validate_settings(settings):
    # "formats" is the additive registry section (lane L2 step 4c); a hand-built document without it
    # still validates and falls back to the built-in list through `document_formats`.
    if not isinstance(settings, dict) or set(settings) not in ({"color", "viewer"}, {"color", "viewer", "formats"}):
        raise ValueError("settings must define color and viewer")
    color = settings["color"]
    if not isinstance(color, dict) or set(color) != {"config", "working_space", "display", "view"}:
        raise ValueError("settings.color is malformed")
    # This release ships one self-contained ACES pipeline. Persisting these fields now gives the
    # project format a clean extension point for user OCIO configs without pretending arbitrary
    # configs are already supported.
    if color["config"] != DEFAULT_SETTINGS["color"]["config"]:
        raise ValueError("Unsupported OCIO config")
    if color["working_space"] != "ACEScg":
        raise ValueError("Working space must be ACEScg")
    if color["display"] != "sRGB - Display":
        raise ValueError("Unsupported display")
    if color["view"] not in ("sRGB", "ACES 2.0", "Linear"):
        raise ValueError("Unsupported display view")
    viewer = settings["viewer"]
    optional = set(VIEWER_STATE_KEYS)
    if (not isinstance(viewer, dict) or "background" not in viewer
            or set(viewer) - {"background", *VIEWER_EXTRA_KEYS} not in (set(), optional)):
        raise ValueError("settings.viewer is malformed")
    if "look" in viewer:
        _validate_look(viewer["look"])
    if "roi" in viewer:
        _validate_roi(viewer["roi"])
    if "proxy" in viewer and (type(viewer["proxy"]) is not int or viewer["proxy"] not in VIEWER_PROXY_TIERS):
        raise ValueError(f"settings.viewer.proxy must be one of {list(VIEWER_PROXY_TIERS)}")
    if "masks" in viewer:
        _validate_masks(viewer["masks"])
    if viewer["background"] not in ("black", "checker"):
        raise ValueError("Viewer background must be black or checker")
    if "inputs" in viewer:
        inputs = viewer["inputs"]
        if (not isinstance(inputs, list) or len(inputs) != VIEWER_INPUT_COUNT
                or any(value is not None and type(value) is not str for value in inputs)):
            raise ValueError(f"settings.viewer.inputs must list {VIEWER_INPUT_COUNT} node IDs or nulls")
        for name in ("active", "b"):
            value = viewer[name]
            if not (value is None and name == "b") and (
                    type(value) is not int or not 1 <= value <= VIEWER_INPUT_COUNT):
                raise ValueError(f"settings.viewer.{name} must be an input number from 1 to {VIEWER_INPUT_COUNT}")
        if viewer["compare"] not in COMPARE_MODES:
            raise ValueError(f"settings.viewer.compare must be one of {list(COMPARE_MODES)}")
    if "formats" in settings:
        formats = settings["formats"]
        if not isinstance(formats, dict) or len(formats) > FORMAT_COUNT_LIMIT:
            raise ValueError(f"settings.formats must be an object of at most {FORMAT_COUNT_LIMIT} formats")
        for name, entry in formats.items():
            _validate_format_entry(name, entry)


def validate_time(time):
    """The document's frame range. Kept separate so the UI can validate an edit before applying it."""
    if not isinstance(time, dict) or set(time) != set(DEFAULT_TIME):
        raise ValueError("Document time must define first, last, current and fps")
    for name, default in DEFAULT_TIME.items():
        value = time[name]
        if type(value) not in (float, int) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"time.{name} must be a finite number")
        if type(default) is int and type(value) is not int:
            raise ValueError(f"time.{name} must be an integer")
        lo, hi = TIME_LIMITS[name]
        if not lo <= value <= hi:
            raise ValueError(f"time.{name} must be between {lo} and {hi}")
    if time["last"] < time["first"]:
        raise ValueError("time.last must not precede time.first")
    if not time["first"] <= time["current"] <= time["last"]:
        raise ValueError("time.current must fall inside the frame range")


def validate(doc):
    if not isinstance(doc, dict) or set(doc) != {"version", "nodes", "view", "time", "animation", "settings", "node_data", "expressions", "references"} or doc["version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported or malformed NodeBased document")
    validate_time(doc["time"])
    validate_settings(doc["settings"])
    nodes = doc["nodes"]
    if not isinstance(nodes, dict) or len(nodes) > 1000:
        raise ValueError("Document must contain at most 1000 nodes")
    if doc["view"] is not None and doc["view"] not in nodes:
        raise ValueError("Viewer target does not exist")
    viewer_inputs = doc["settings"]["viewer"].get("inputs")
    if viewer_inputs is not None:
        if any(value is not None and value not in nodes for value in viewer_inputs):
            raise ValueError("settings.viewer.inputs names a missing node")
        if viewer_inputs[doc["settings"]["viewer"]["active"] - 1] != doc["view"]:
            raise ValueError("The active viewer input must be the viewed node")
    references = doc["references"]
    if not isinstance(references, list) or len(references) > 1000:
        raise ValueError("references must be a list of at most 1000 node IDs")
    if any(type(node_id) is not str for node_id in references):
        raise ValueError("references must contain only node IDs as strings")
    if len(set(references)) != len(references):
        raise ValueError("references must not contain duplicate node IDs")
    if any(node_id not in nodes for node_id in references):
        raise ValueError("references contains a missing node ID")
    for key, node in nodes.items():
        if not isinstance(key, str) or not key or len(key) > 128:
            raise ValueError("Invalid node ID")
        if (not isinstance(node, dict)
                or not NODE_KEYS <= set(node) <= NODE_KEYS | OPTIONAL_NODE_KEYS):
            raise ValueError("Malformed node")
        if "label" in node and (not isinstance(node["label"], str) or not node["label"]
                                or len(node["label"]) > NODE_LABEL_LIMIT):
            raise ValueError(f"label must be a non-empty string of at most {NODE_LABEL_LIMIT} characters")
        if "thumbnail" in node and type(node["thumbnail"]) is not bool:
            raise ValueError("thumbnail must be boolean")
        kind = node["type"]
        if kind not in SPECS:
            raise ValueError(f"Unknown node type: {kind}")
        spec = SPECS[kind]
        if not isinstance(node["name"], str) or not 1 <= len(node["name"]) <= 128:
            raise ValueError("Node name must contain 1–128 characters")
        if type(node["disabled"]) is not bool:
            raise ValueError("disabled must be boolean")
        if not isinstance(node["pos"], list) or len(node["pos"]) != 2 or any(type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1e6 for v in node["pos"]):
            raise ValueError("Invalid node position")
        if not isinstance(node["params"], dict) or set(node["params"]) != set(spec["params"]):
            raise ValueError(f"Invalid parameters for {kind}")
        for name, default in spec["params"].items():
            value = node["params"][name]
            if kind == "Reformat" and name == "format":
                # A Reformat may name any format in the document registry, so the static CHOICES
                # list (the built-ins, kept for discovery) is not the whole set of valid values.
                if value != "Custom" and value not in document_formats(doc):
                    raise ValueError(f"Invalid {name}: {value}")
                continue
            if name in CHOICES and value not in CHOICES[name]:
                raise ValueError(f"Invalid {name}: {value}")
            if isinstance(default, str):
                if not isinstance(value, str) or len(value) > 32768:
                    raise ValueError(f"{name} must be a path string")
            else:
                if type(value) not in (float, int) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite number")
                if type(default) is int and type(value) is not int:
                    raise ValueError(f"{name} must be an integer")
                lo, hi = LIMITS[name]
                if not lo <= value <= hi:
                    raise ValueError(f"{name} must be between {lo} and {hi}")
        # Inputs cover both required slots (in SPECS[kind]["inputs"]) and optional slots (in
        # SPECS[kind].get("optional_inputs")). Required must be wired before evaluation; optional
        # may be None and acts as identity (full opacity mask, no input selection).
        expected_inputs = set(spec["inputs"]) | set(spec.get("optional_inputs", []))
        if not isinstance(node["inputs"], dict) or set(node["inputs"]) != expected_inputs:
            raise ValueError(f"Invalid inputs for {kind}")
        for slot, source in node["inputs"].items():
            if source is not None and (not isinstance(source, str) or source not in nodes):
                raise ValueError(f"Input {slot!r} references a missing node")
            if source is not None:
                expected_type = INPUT_TYPES.get(slot, ("image",))
                actual_type = OUTPUT_TYPES.get(nodes[source]["type"], "image")
                if actual_type not in expected_type:
                    raise ValueError(f"Input {slot!r} on {node['name']!r} expects {' or '.join(expected_type)}, "
                                     f"got {actual_type} from {nodes[source]['name']!r}")
    shapes.validate_node_data(doc["node_data"], nodes)
    expr.validate_expressions(doc["expressions"], nodes, lambda kind: SPECS[kind]["params"])
    # One driver per parameter. A curve and an expression on the same knob is rejected rather than
    # resolved by a precedence rule, because a silent precedence rule is the bug where an artist
    # keys a knob, sees nothing move, and has no way to find out why.
    for node_id, params in doc["expressions"].items():
        keyed = set((doc["animation"]["curves"].get(node_id) or {}))
        clash = sorted(keyed & set(params))
        if clash:
            raise ValueError(f"{node_id}: {', '.join(clash)} has both an animation curve and an "
                             f"expression; a parameter takes exactly one of the two")
    # Iterative topological walk avoids recursion-limit crashes on long graphs.
    pending = {key: sum(v is not None for v in n["inputs"].values()) for key, n in nodes.items()}
    children = {key: [] for key in nodes}
    for key, node in nodes.items():
        for source in node["inputs"].values():
            if source is not None:
                children[source].append(key)
    ready = [key for key, count in pending.items() if count == 0]
    count = 0
    while ready:
        key = ready.pop()
        count += 1
        for child in children[key]:
            pending[child] -= 1
            if pending[child] == 0:
                ready.append(child)
    if count != len(nodes):
        raise ValueError("Image graphs cannot contain cycles; AI loops will use structured tasks")
    # Animation section is always present post-v5: {"curves": {node_id: {param: curve}}}. Validate
    # structural shape plus that each curve points at a real node and a numeric parameter on it,
    # and that the curve itself satisfies validate_curve().
    animation = doc["animation"]
    if not isinstance(animation, dict) or set(animation) != {"curves"}:
        raise ValueError("animation must define exactly 'curves'")
    curves_root = animation["curves"]
    if not isinstance(curves_root, dict):
        raise ValueError("animation.curves must be an object")
    # Local import to avoid a circular dependency at module import time.
    from .animation import validate_curve as _validate_curve
    for node_id, params_curves in curves_root.items():
        if node_id not in nodes:
            raise ValueError(f"animation.curves references missing node {node_id!r}")
        if not isinstance(params_curves, dict):
            raise ValueError(f"animation.curves[{node_id!r}] must be an object")
        node = nodes[node_id]
        spec_params = SPECS[node["type"]]["params"]
        for param_name, curve in params_curves.items():
            if not isinstance(param_name, str) or not param_name:
                raise ValueError("animation curve key must be a non-empty string")
            if param_name not in spec_params:
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: unknown parameter for "
                    f"{node['type']!r}")
            spec_default = spec_params[param_name]
            if not isinstance(spec_default, (int, float)) or isinstance(spec_default, bool):
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: only numeric parameters can "
                    f"be animated (this is a {type(spec_default).__name__})")
            if not isinstance(curve, dict):
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: curve must be an object")
            try:
                _validate_curve(curve)
            except Exception as error:
                raise ValueError(
                    f"animation.curves[{node_id!r}].{param_name}: {error}") from error


def atomic_save(path, doc):
    path = Path(path).expanduser().resolve()
    validate(doc)
    # Store read paths relative to the project for portable folder trees.
    portable = copy.deepcopy(doc)
    for node in portable["nodes"].values():
        if node["type"] == "Read" and node["params"]["path"]:
            try:
                node["params"]["path"] = os.path.relpath(node["params"]["path"], path.parent)
            except ValueError:  # Windows different drive.
                pass
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(portable, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load_document(path):
    path = Path(path).expanduser().resolve()
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError("Project exceeds 4 MiB document limit")
    doc = upgrade_document(json.loads(path.read_text(encoding="utf-8")))
    validate(doc)
    for node in doc["nodes"].values():
        if node["type"] == "Read" and node["params"]["path"]:
            node["params"]["path"] = str((path.parent / node["params"]["path"]).resolve())
    return doc


class Dispatcher:
    """Each edit/batch is validated atomically and takes one undo slot."""
    def __init__(self, document=None):
        self.document = upgrade_document(document or empty_document())
        validate(self.document)
        self.undo_stack = []
        self.redo_stack = []
        self.revision = 0

    def execute(self, request):
        if not isinstance(request, dict):
            raise ValueError("Command must be an object")
        op = request.get("op")
        if "if_revision" in request:
            expected = request["if_revision"]
            if type(expected) is not int:
                raise ValueError("if_revision must be an integer")
            if expected != self.revision:
                raise ValueError(f"if_revision {expected} is stale; current revision is {self.revision}")
        if op == "describe":
            # Local import keeps the top-level Dispatcher import cycle-free.
            from .animation import CURVE_INTERPOLATIONS as _CURVE_INTERPOLATIONS, FRAME_LIMITS as _FRAME_LIMITS
            return {"protocol": 1, "nodes": copy.deepcopy(SPECS), "limits": LIMITS, "choices": CHOICES,
                    "time": copy.deepcopy(self.document["time"]), "time_limits": TIME_LIMITS,
                    "sequence_patterns": ["printf (plate.%04d.exr)", "hash (plate.####.exr)", "still (plate.exr)"],
                    "animation": {
                        "interpolations": list(_CURVE_INTERPOLATIONS),
                        "frame_limits": list(_FRAME_LIMITS),
                        "shape": {"interpolation": "constant | linear",
                                  "keys": [{"frame": "int", "value": "number"}]},
                        "operations": ["set_key", "delete_key", "clear_curve"],
                        "set_key": {"id": "string (node id)", "param": "string (numeric parameter name)",
                                     "frame": "int (timeline frame)", "value": "number (finite)",
                                     "interpolation": "constant | linear (optional, default 'linear')"},
                        "delete_key": {"id": "string", "param": "string", "frame": "int"},
                        "clear_curve": {"id": "string", "param": "string"}},
                    "settings": copy.deepcopy(self.document["settings"]),
                    "artifact_types": {kind: artifact_type(kind) for kind in SPECS},
                    # Structured per-node payloads (schema v8). An agent discovers which node types
                    # carry one, and the envelope every time-varying number inside it uses — the
                    # same curve envelope the "animation" block above describes.
                    "node_data": {"payloads": dict(shapes.NODE_DATA_SCHEMA),
                                  "shape_modes": list(shapes.SHAPE_MODES),
                                  "scalar_limits": {k: list(v) for k, v in shapes.SHAPE_LIMITS.items()},
                                  "point_fields": list(shapes.POINT_FIELDS),
                                  "set_shapes": {"id": "string (Roto node id)",
                                                 "shapes": "[{name, mode, opacity, feather, points}]"},
                                  "set_tracks": {"id": "string (Tracker node id)",
                                                 "tracks": "[{name, enabled, x, y}]"}},
                    # Knob expressions (schema v9). A numeric parameter may carry a curve or an
                    # expression, never both; the document is rejected if it carries both.
                    "expressions": {
                        "reference": 'knob("node id", "param"), or node_id.param when the id is a '
                                     'valid identifier. References are by node id, so renaming a '
                                     'node never breaks a link.',
                        "names": [expr.FRAME_NAME] + sorted(expr.CONSTANTS),
                        "functions": sorted(expr.FUNCTIONS),
                        "max_length": expr.MAX_EXPRESSION_LENGTH,
                        "set_expression": {"id": "string", "param": "string",
                                           "expression": "string"},
                        "clear_expression": {"id": "string", "param": "string"}},
                    "references": {"current": list(self.document["references"]),
                                   "operation": {"id": "string (existing node id)",
                                                 "value": "boolean (true appends, false removes)"}},
                    "operations": ["describe", "inspect", "create", "set", "connect", "move", "rename", "label", "thumbnail", "disable", "delete", "reference", "view", "viewer_input", "viewer_compare", "viewer_look", "viewer_roi", "viewer_proxy", "viewer_mask", "time", "settings", "format", "set_key", "delete_key", "clear_curve", "set_shapes", "set_tracks", "set_expression", "clear_expression", "batch", "undo", "redo", "save", "load"]}
        if op == "inspect":
            return {"revision": self.revision, "references": list(self.document["references"]),
                    "document": copy.deepcopy(self.document)}
        if op == "save":
            atomic_save(request["path"], self.document)
            return {"path": str(Path(request["path"]).resolve())}
        if op in ("undo", "redo"):
            source, target = (self.undo_stack, self.redo_stack) if op == "undo" else (self.redo_stack, self.undo_stack)
            if source:
                target.append(self.document)
                self.document = source.pop()
                self.revision += 1
            return {"revision": self.revision}
        draft = copy.deepcopy(self.document)
        if op == "load":
            draft = load_document(request["path"])
            result = {}
        elif op == "batch":
            commands = request.get("commands")
            if not isinstance(commands, list) or len(commands) > 1000:
                raise ValueError("batch requires at most 1000 commands")
            result = [self._edit(draft, command) for command in commands]
        else:
            result = self._edit(draft, request)
        validate(draft)
        if draft != self.document:
            # Transport playback updates the persisted playhead through the same validated command
            # boundary, but must not fill the artist's undo history once per frame.
            transient_time = op == "time" and request.get("transient") is True
            if not transient_time:
                self.undo_stack.append(self.document)
                self.undo_stack = self.undo_stack[-100:]
                self.redo_stack.clear()
            self.document = draft
            self.revision += 1
        return {"revision": self.revision, "result": result}

    def _edit(self, doc, cmd):
        op = cmd["op"]
        nodes = doc["nodes"]
        if op == "create":
            kind = cmd["type"]
            if kind not in SPECS:
                raise ValueError(f"Unknown node type: {kind}")
            key = cmd.get("id", uuid.uuid4().hex[:12])
            if key in nodes:
                raise ValueError("Node ID already exists")
            params = {**SPECS[kind]["params"], **cmd.get("params", {})}
            if kind == "Read" and params["path"]:
                params["path"] = str(Path(params["path"]).expanduser().resolve())
            if kind == "Reformat" and "format" in cmd.get("params", {}):
                # Only when the caller actually named a format: an explicit width/height passed
                # alongside a default, unmentioned "format" must not be clobbered back to preset.
                _resolve_reformat_format(params, doc)
            nodes[key] = {"type": kind, "name": cmd.get("name", kind), "params": params,
                          "inputs": {slot: None for slot in
                                     list(SPECS[kind]["inputs"]) + list(SPECS[kind].get("optional_inputs", []))},
                          "pos": cmd.get("pos", [0, 0]), "disabled": False}
            return {"id": key}
        if op == "view":
            target = cmd.get("id")
            state = viewer_state(doc)
            _apply_view(doc, target)
            # Viewing a node fills the input the viewer is on, so `view` and the input strip
            # never disagree about what is being shown.
            state["inputs"][state["active"] - 1] = target
            _store_viewer_state(doc, state)
            return {}
        if op == "viewer_input":
            slot = _viewer_slot(cmd.get("slot"))
            target = cmd.get("id")
            if target is not None and target not in nodes:
                raise ValueError(f"viewer_input: unknown node id {target!r}")
            activate = cmd.get("activate", True)
            if type(activate) is not bool:
                raise ValueError("viewer_input: activate must be boolean")
            state = viewer_state(doc)
            state["inputs"][slot - 1] = target
            if activate:
                state["active"] = slot
            if state["active"] == slot:
                _apply_view(doc, target)
            _store_viewer_state(doc, state)
            return {"inputs": list(state["inputs"]), "active": state["active"]}
        if op == "viewer_compare":
            state = viewer_state(doc)
            if "b" in cmd:
                state["b"] = None if cmd["b"] is None else _viewer_slot(cmd["b"], "b")
            if "mode" in cmd:
                if cmd["mode"] not in COMPARE_MODES:
                    raise ValueError(f"viewer_compare: mode must be one of {list(COMPARE_MODES)}")
                state["compare"] = cmd["mode"]
            _store_viewer_state(doc, state)
            return {"b": state["b"], "compare": state["compare"]}
        if op == "viewer_look":
            look = viewer_look(doc)
            for name in VIEWER_LOOK_DEFAULT:
                if name in cmd:
                    look[name] = cmd[name]
            _validate_look(look)
            if look["display"] != "Project view":
                from .color import viewer_displays
                if look["display"] not in viewer_displays():
                    raise ValueError(f"viewer_look: display must be one of {['Project view', *viewer_displays()]}")
            look["gain"], look["gamma"] = float(look["gain"]), float(look["gamma"])
            if look == VIEWER_LOOK_DEFAULT:
                doc["settings"]["viewer"].pop("look", None)
            else:
                doc["settings"]["viewer"]["look"] = look
            return dict(look)
        if op == "viewer_roi":
            roi = viewer_roi(doc)
            for name in ("on", "rect"):
                if name in cmd:
                    roi[name] = cmd[name]
            if isinstance(roi["rect"], (list, tuple)):
                roi["rect"] = [float(value) if type(value) in (int, float) else value for value in roi["rect"]]
            _validate_roi(roi)
            if roi == VIEWER_ROI_DEFAULT:
                doc["settings"]["viewer"].pop("roi", None)
            else:
                doc["settings"]["viewer"]["roi"] = roi
            return dict(roi)
        if op == "viewer_proxy":
            tier = cmd.get("tier")
            if type(tier) is not int or tier not in VIEWER_PROXY_TIERS:
                raise ValueError(f"viewer_proxy: tier must be one of {list(VIEWER_PROXY_TIERS)}")
            if tier == 1:
                doc["settings"]["viewer"].pop("proxy", None)
            else:
                doc["settings"]["viewer"]["proxy"] = tier
            return {"tier": tier}
        if op == "viewer_mask":
            masks = viewer_masks(doc)
            for name in VIEWER_MASKS_DEFAULT:
                if name in cmd:
                    masks[name] = cmd[name]
            _validate_masks(masks)
            if masks == VIEWER_MASKS_DEFAULT:
                doc["settings"]["viewer"].pop("masks", None)
            else:
                doc["settings"]["viewer"]["masks"] = masks
            return dict(masks)
        if op == "reference":
            key = cmd.get("id")
            if not isinstance(key, str) or key not in nodes:
                raise ValueError(f"reference: unknown node id {key!r}")
            value = cmd.get("value")
            if type(value) is not bool:
                raise ValueError("reference: value must be boolean")
            references = doc["references"]
            if value and key not in references:
                references.append(key)
            elif not value and key in references:
                references.remove(key)
            return {"references": list(references)}
        if op in ("set_key", "delete_key", "clear_curve"):
            return self._animation_edit(doc, cmd)
        if op in ("set_expression", "clear_expression"):
            return self._expression_edit(doc, cmd)
        if op == "time":
            # Scrubbing the playhead and re-ranging the comp are both document edits, so they are
            # undoable and reach an attached agent through the same validated boundary as any other
            # change. Clamp current into the range so setting first/last cannot strand the playhead.
            time = dict(doc["time"])
            for name in DEFAULT_TIME:
                if name in cmd:
                    time[name] = cmd[name]
            if "current" not in cmd:
                time["current"] = min(max(time["current"], time["first"]), time["last"])
            validate_time(time)
            doc["time"] = time
            return dict(time)
        if op == "format":
            return _edit_format(doc, cmd)
        if op == "settings":
            changes = cmd.get("settings")
            if not isinstance(changes, dict):
                raise ValueError("settings operation requires a settings object")
            unknown = set(changes) - {"color", "viewer"}
            if unknown:
                raise ValueError(f"Unknown settings groups: {sorted(unknown)}")
            for group, values in changes.items():
                if not isinstance(values, dict):
                    raise ValueError(f"settings.{group} must be an object")
                unknown_fields = set(values) - set(doc["settings"][group])
                if unknown_fields:
                    raise ValueError(f"Unknown settings.{group} fields: {sorted(unknown_fields)}")
                doc["settings"][group].update(values)
            return copy.deepcopy(doc["settings"])
        key = cmd["id"]
        node = nodes[key]
        if op == "set":
            name = cmd["param"]
            value = cmd["value"]
            if node["type"] == "Read" and name == "path" and isinstance(value, str) and value:
                value = str(Path(value).expanduser().resolve())
            node["params"][name] = value
            if node["type"] == "Reformat" and name == "format":
                _resolve_reformat_format(node["params"], doc)
        elif op == "connect":
            node["inputs"][cmd["input"]] = cmd.get("source")
        elif op == "move":
            node["pos"] = cmd["pos"]
        elif op == "rename":
            node["name"] = cmd["name"]
        elif op == "label":
            value = cmd.get("value")
            if not isinstance(value, str):
                raise ValueError("label: value must be a string")
            if value:
                node["label"] = value
            else:
                node.pop("label", None)
        elif op == "thumbnail":
            value = cmd.get("value")
            if type(value) is not bool:
                raise ValueError("thumbnail: value must be boolean")
            if value == (node["type"] in DEFAULT_THUMBNAIL_TYPES):
                node.pop("thumbnail", None)
            else:
                node["thumbnail"] = value
        elif op in ("set_shapes", "set_tracks"):
            # Whole-payload replacement, validated by `validate` like any other edit and taking one
            # undo slot. There is no per-key op: keying a single point goes through the animation
            # curve on that point's scalar, reusing `animation.merge_key` rather than growing a
            # second key-insert path.
            slot = "shapes" if op == "set_shapes" else "tracks"
            if shapes.payload_slot(node["type"]) != slot:
                raise ValueError(f"{op}: {node['type']} nodes do not carry {slot}")
            items = cmd[slot]
            if not isinstance(items, list):
                raise ValueError(f"{op}: {slot!r} must be a list")
            # An empty payload is stored as an absent entry so "no shapes" has one representation
            # in the document and two comps that look identical also serialize identically.
            if items:
                doc["node_data"][key] = {slot: copy.deepcopy(items)}
            else:
                doc["node_data"].pop(key, None)
            return {slot: len(items)}
        elif op == "disable":
            # A node with nothing `bypass_slot` can name (a pure generator: Read, Constant,
            # Checker, Roto) has no meaningful "disabled" behaviour and is refused outright. A Draw
            # node (Ramp/Radial/Rectangle/Noise/Text) has empty required `inputs` too, exactly like
            # those, but DRAW_KINDS gives `bypass_slot` its optional "image" to pass through (or a
            # transparent frame when that is unwired), so it is bypassable like any other node.
            if bypass_slot(node) is None:
                raise ValueError("Source nodes cannot be bypassed")
            node["disabled"] = cmd["value"]
        elif op == "delete":
            # An expression on *another* node reading this one cannot be silently dropped and
            # cannot be left dangling either, so the delete is refused and names the links. Both
            # alternatives are worse: dropping loses work the artist cannot get back, and leaving
            # it makes the document fail its own validation on save.
            dependents = sorted(
                f"{other}.{name}"
                for other, params in doc["expressions"].items() if other != key
                for name, text in params.items()
                if any(target == key for target, _ in expr.parse(text).references))
            if dependents:
                raise ValueError(
                    f"delete: {key!r} is referenced by the expression on "
                    f"{', '.join(dependents)}; clear those expressions first")
            del nodes[key]
            for other in nodes.values():
                other["inputs"] = {slot: None if value == key else value for slot, value in other["inputs"].items()}
            state = viewer_state(doc)
            if doc["view"] == key:
                doc["view"] = None
            state["inputs"] = [None if value == key else value for value in state["inputs"]]
            _store_viewer_state(doc, state)
            # Atomically drop any curves targeting the deleted node. Without this, validate()
            # would reject the post-delete document for referencing a missing node. The undo
            # stack already holds a deep copy of the pre-delete document (including the node's
            # animation entry), so undo restores both the node and its curves automatically.
            doc["animation"]["curves"].pop(key, None)
            # node_data is keyed by node id for the same reason and needs the same atomic clear.
            doc["node_data"].pop(key, None)
            # The deleted node's own expressions go with it; inbound references were refused above.
            doc["expressions"].pop(key, None)
            if key in doc["references"]:
                doc["references"].remove(key)
        else:
            raise ValueError(f"Unknown edit operation: {op}")
        return {}

    def _animation_edit(self, doc, cmd):
        """Atomic, undoable edits to doc["animation"]. The node's stored params are never
        touched: a curve is an evaluation-time override only. See ``nodebased/animation.py``."""
        from .animation import (CURVE_INTERPOLATIONS as _CURVE_INTERPOLATIONS,
                                  coerce_value_for_param as _coerce,
                                  drop_key as _drop_key, merge_key as _merge_key,
                                  validate_curve as _validate_curve)
        op = cmd["op"]
        node_id = cmd.get("id")
        param = cmd.get("param")
        if not isinstance(node_id, str) or node_id not in doc["nodes"]:
            raise ValueError(f"animation edit {op!r}: unknown node id {node_id!r}")
        if not isinstance(param, str) or not param:
            raise ValueError(f"animation edit {op!r}: 'param' must be a non-empty string")
        node = doc["nodes"][node_id]
        spec_params = SPECS[node["type"]]["params"]
        if param not in spec_params:
            raise ValueError(f"animation edit {op!r}: {node['type']!r} has no parameter {param!r}")
        spec_default = spec_params[param]
        if not isinstance(spec_default, (int, float)) or isinstance(spec_default, bool):
            raise ValueError(
                f"animation edit {op!r}: parameter {param!r} is not numeric "
                f"({type(spec_default).__name__})")
        if param in (doc["expressions"].get(node_id) or {}):
            raise ValueError(
                f"animation edit {op!r}: {param!r} is driven by an expression; clear the "
                f"expression first, because a parameter takes a curve or a formula, not both")
        curves_root = doc["animation"]["curves"]
        node_curves = curves_root.setdefault(node_id, {})
        curve = node_curves.get(param)
        if op == "set_key":
            interpolation = cmd.get("interpolation", curve["interpolation"] if curve else "linear")
            if interpolation not in _CURVE_INTERPOLATIONS:
                raise ValueError(
                    f"animation set_key: interpolation must be one of {list(_CURVE_INTERPOLATIONS)}")
            value = cmd["value"]
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError("animation set_key: 'value' must be a number")
            import math as _math
            if not _math.isfinite(value):
                raise ValueError("animation set_key: 'value' must be a finite number")
            # Range check using the spec's limit when present. The set_key layer rejects out-of-
            # range values explicitly so the caller gets a clear error rather than silent clamping.
            if param in LIMITS:
                lo, hi = LIMITS[param]
                if value < lo or value > hi:
                    raise ValueError(
                        f"animation set_key: value {value} outside [{lo}, {hi}] for {param!r}")
            frame = cmd["frame"]
            if type(frame) is not int or isinstance(frame, bool):
                raise ValueError("animation set_key: 'frame' must be an integer")
            base = {"interpolation": interpolation, "keys": []} if curve is None else curve
            new_curve = _merge_key(base, frame, value)
            _validate_curve(new_curve)
            node_curves[param] = new_curve
            return {"id": node_id, "param": param, "frame": frame, "value": float(value),
                    "interpolation": interpolation}
        if op == "delete_key":
            frame = cmd["frame"]
            if type(frame) is not int or isinstance(frame, bool):
                raise ValueError("animation delete_key: 'frame' must be an integer")
            if curve is None:
                raise ValueError(f"animation delete_key: no curve on {node_id!r}.{param!r}")
            new_curve = _drop_key(curve, frame)
            if new_curve is None:
                del node_curves[param]
                if not node_curves:
                    del curves_root[node_id]
                return {"id": node_id, "param": param, "frame": frame, "removed": True}
            node_curves[param] = new_curve
            return {"id": node_id, "param": param, "frame": frame, "remaining_keys": len(new_curve["keys"])}
        if op == "clear_curve":
            if param not in node_curves:
                raise ValueError(f"animation clear_curve: no curve on {node_id!r}.{param!r}")
            del node_curves[param]
            if not node_curves:
                del curves_root[node_id]
            return {"id": node_id, "param": param, "cleared": True}
        raise ValueError(f"Unknown animation operation: {op}")

    def _expression_edit(self, doc, cmd):
        """Atomic, undoable edits to doc["expressions"].

        Like a curve, an expression never touches the node's stored params -- it is an
        evaluation-time override, so clearing one restores the number the artist last typed rather
        than leaving whatever the formula happened to produce.

        The whole edited section is re-validated before the edit is accepted, so a cycle is
        rejected by the op that would create it. Catching it here rather than at save time means
        the error names the link the artist just made, while they are still looking at it.
        """
        op = cmd["op"]
        node_id = cmd.get("id")
        param = cmd.get("param")
        if not isinstance(node_id, str) or node_id not in doc["nodes"]:
            raise ValueError(f"expression edit {op!r}: unknown node id {node_id!r}")
        if not isinstance(param, str) or not param:
            raise ValueError(f"expression edit {op!r}: 'param' must be a non-empty string")
        node = doc["nodes"][node_id]
        spec_default = SPECS[node["type"]]["params"].get(param)
        if spec_default is None:
            raise ValueError(f"expression edit {op!r}: {node['type']!r} has no parameter {param!r}")
        if isinstance(spec_default, str) or isinstance(spec_default, bool):
            raise ValueError(f"expression edit {op!r}: parameter {param!r} is not numeric "
                             f"({type(spec_default).__name__})")
        root = doc["expressions"]

        if op == "clear_expression":
            if param not in (root.get(node_id) or {}):
                raise ValueError(f"clear_expression: no expression on {node_id!r}.{param!r}")
            del root[node_id][param]
            if not root[node_id]:
                del root[node_id]
            return {"id": node_id, "param": param, "cleared": True}

        text = cmd.get("expression")
        if not isinstance(text, str):
            raise ValueError("set_expression: 'expression' must be a string")
        if param in (doc["animation"]["curves"].get(node_id) or {}):
            raise ValueError(f"set_expression: {param!r} is animated; clear the curve first, "
                             f"because a parameter takes a curve or a formula, not both")
        candidate = {key: dict(value) for key, value in root.items()}
        candidate.setdefault(node_id, {})[param] = text
        # Raises ValueError on a bad formula, a dangling reference or a cycle, before the document
        # is touched at all.
        expr.validate_expressions(candidate, doc["nodes"], lambda kind: SPECS[kind]["params"])
        root.setdefault(node_id, {})[param] = text
        return {"id": node_id, "param": param, "expression": text,
                "references": [f"{target}.{name}" for target, name in expr.parse(text).references]}


def demo_document():
    d = Dispatcher()
    d.execute({"op": "batch", "commands": [
        {"op": "create", "id": "plate", "type": "Checker", "name": "Checker · procedural plate", "pos": [-110, -300]},
        {"op": "create", "id": "grade", "type": "Grade", "pos": [-110, -130], "params": {"exposure": 0.35}},
        {"op": "connect", "id": "grade", "input": "image", "source": "plate"},
        {"op": "create", "id": "wash", "type": "Constant", "name": "Constant · blue wash", "pos": [-400, -150], "params": {"alpha": 0.22}},
        {"op": "create", "id": "merge", "type": "Merge", "pos": [-110, 10]},
        {"op": "connect", "id": "merge", "input": "A", "source": "wash"},
        {"op": "connect", "id": "merge", "input": "B", "source": "grade"},
        {"op": "create", "id": "viewer", "type": "Viewer", "pos": [-110, 130]},
        {"op": "connect", "id": "viewer", "input": "image", "source": "merge"},
        {"op": "view", "id": "viewer"}]})
    return d.document
