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
from . import shapes

# Parameter schemas are also consumed by the inspector and agent discovery.
# Filter nodes accept an optional "mask" image (alpha gates where the filter applies) and a
# per-node "mix" (blend between original input and filtered output). The mask slot is listed in
# "optional_inputs" rather than "inputs" so a node validates without it wired — the evaluator
# treats None there as full opacity (M.a = 1).
IMAGE_FILTER_KINDS = ("Grade", "ColorCorrect", "Blur", "Transform", "Crop")

# Kinds that honour the optional-mask + mix contract. IMAGE_FILTER_KINDS is frozen history — the
# v3 -> v4 upgrade is written against it — so a kind that adopts the contract later joins this
# list instead, which is what the inspector and the evaluator read.
MASK_MIX_KINDS = IMAGE_FILTER_KINDS + ("Tracker", "Invert", "Clamp", "Multiply", "Add", "Gamma",
                                       "Saturation", "Erode", "Dilate", "Median", "Sharpen", "Glow",
                                       "Mirror", "Keyer", "HueKeyer")

# Two-input A/B kinds sharing Merge's bypass and windowing convention: bypass passes B (the
# background), or A when B is unwired; the union of A's and B's data windows is the output; an
# optional mask aligns to B's display window. See `bypass_slot` and `imaging._windowed_kernel`.
MERGE_LIKE_KINDS = ("Merge", "Dissolve", "Keymix", "Copy", "ChannelMerge")

# Draw-menu generators: own format (width/height), plus an optional "image" input the shape is
# composited over and an optional "mask". Bypassing one passes that optional image through (or a
# transparent format-sized frame when it is unwired) rather than the empty-slot behaviour a pure
# generator like Constant/Checker/Roto gets, because a Draw node's whole point is to sit inline in
# a chain over an existing plate. See `bypass_slot` and `imaging.Evaluator._windowed_kernel`.
DRAW_KINDS = ("Ramp", "Radial", "Rectangle", "Noise", "Text")

# The version `upgrade_document` migrates to and `validate` accepts. Tests and callers should refer
# to this rather than hard-coding a number, so a schema bump does not spray stale literals.
SCHEMA_VERSION = 12
# Node-tab fields (Nuke's "Node" tab). Both are optional on a node and absent means default, so
# a comp has one serialized form: a node only carries them once an artist changed them.
NODE_LABEL_LIMIT = 1024
# Sources are where a postage stamp tells you something; on a filter it mostly repeats the input.
DEFAULT_THUMBNAIL_TYPES = ("Read", "Constant", "Checker")

# Connection typing is deliberately small and explicit.  A Render3D node is the only bridge
# from scene/camera values to the existing image graph; this prevents a malformed graph from
# failing much later inside a renderer.
# The full Nuke-style transform: T(translate) @ T(pivot) @ R(rot_order) @ S * uscale @ T(-pivot).
_XFORM = {"tx": 0.0, "ty": 0.0, "tz": 0.0, "rx": 0.0, "ry": 0.0, "rz": 0.0,
          "sx": 1.0, "sy": 1.0, "sz": 1.0, "uscale": 1.0, "rot_order": "XYZ",
          "pivot_x": 0.0, "pivot_y": 0.0, "pivot_z": 0.0}
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
    "Mirror": {"inputs": ["image"], "optional_inputs": ["mask"],
               "params": {"flip_x": 0, "flip_y": 0, "mix": 1.0}},
    "Transform": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"translate_x": 0.0, "translate_y": 0.0, "rotate": 0.0,
                                                 "scale": 1.0, "center_x": 0.0, "center_y": 0.0, "filter": "nearest", "mix": 1.0}},
    "Crop": {"inputs": ["image"], "optional_inputs": ["mask"], "params": {"x": 0, "y": 0, "width": 960, "height": 540, "mix": 1.0}},
    "Shuffle": {"inputs": ["image"], "params": {"red_from": "R", "green_from": "G", "blue_from": "B", "alpha_from": "A"}},
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
    "Premult": {"inputs": ["image"], "params": {}},
    "Unpremult": {"inputs": ["image"], "params": {}},
    "Dot": {"inputs": ["input"], "params": {}},
    "Switch": {"inputs": ["0", "1"], "params": {"which": 0}},
    "Viewer": {"inputs": ["image"], "params": {}},
    # Write is where real image output lives, as in Nuke: the node states the destination and the
    # format, and rendering it is an explicit action rather than a side effect of looking at a
    # frame. It passes its input through unchanged, so a Write parked mid-branch never alters the
    # comp downstream of it. "file_type" Auto takes the extension on "path" at its word.
    "Write": {"inputs": ["image"], "params": {"path": "", "file_type": "Auto", "bit_depth": "half"}},
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
    "Card3D": {"inputs": [], "optional_inputs": ["image"],
               "params": {"card_width": 2.0, "card_height": 2.0, **_XFORM, **_SURFACE}},
    "Cube3D": {"inputs": [], "optional_inputs": ["image"],
               "params": {"cube_size": 2.0, **_XFORM, **_SURFACE}},
    "Sphere3D": {"inputs": [], "optional_inputs": ["image"],
                 "params": {"sphere_radius": 1.0, "segments": 32, **_XFORM, **_SURFACE}},
    "ReadSplat3D": {"inputs": [], "params": {
        "splat_path": "", "splat_orientation": "as_authored", "splat_colorspace": "srgb",
        "splat_sh_degree": 3, "splat_opacity": 1.0, "splat_scale": 1.0, "splat_relight": 0.0,
        "splat_shadow_catch": 0.0, "splat_cast_shadows": "on",
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
                                         "red": 1.0, "green": 1.0, "blue": 1.0, "intensity": 1.0, "shadows": "off"}},
    "Camera3D": {"inputs": [], "params": {"tx": 0.0, "ty": 0.0, "tz": 5.0, "roll": 0.0,
                                          "target_x": 0.0, "target_y": 0.0, "target_z": 0.0,
                                          "fov": 45.0, "near": 0.1, "far": 1000.0}},
    "Project3D": {"inputs": ["image", "camera", "geometry"],
                  "params": {"project_outside": "transparent", "project_backfaces": "project", "project_occlusion": "off"}},
    "WriteGeo3D": {"inputs": ["scene"], "params": {"geo_write_path": ""}},
    "Scene3D": {"inputs": [], "optional_inputs": [f"object{i}" for i in range(8)], "params": dict(_XFORM)},
    "Relight": {"inputs": ["image"], "optional_inputs": ["camera"] + [f"light{i}" for i in range(8)],
                "params": {"red": 0.8, "green": 0.8, "blue": 0.8,
                           "diffuse": 1.0, "specular": 1.0, "mix": 1.0}},
    "Render3D": {"inputs": ["scene", "camera"],
                 "params": {"width": 960, "height": 540, "red": 0.0, "green": 0.0, "blue": 0.0,
                            "alpha": 0.0, "ambient": 0.1, "samples": 2, "render_output": "rgba", "render_backend": "cpu", "render_mode": "raster"}},
}


def bypass_slot(node):
    """The one input slot a bypassed (disabled) node passes through, or None.

    Every evaluation path asks here, so the traversal, the cache digests and the pixels cannot
    disagree about what a bypassed node is. A Merge passes B, its background, as in Nuke: bypassing
    the merge removes what was laid over the main pipe rather than the pipe itself. With B unwired
    it passes A. Project3D passes its geometry. Axis3D passes its object (its only slot, but
    optional, so it is not in SPECS["Axis3D"]["inputs"]). Everything else passes its first declared
    input.
    """
    kind, inputs = node["type"], node["inputs"]
    if kind == "Project3D":
        return "geometry"
    if kind == "Axis3D":
        return "object"
    if kind in MERGE_LIKE_KINDS:
        return "B" if inputs.get("B") is not None or inputs.get("A") is None else "A"
    if kind in DRAW_KINDS:
        return "image"
    slots = SPECS[kind]["inputs"]
    return slots[0] if slots else None


OUTPUT_TYPES = {kind: "image" for kind in SPECS}
GEOMETRY_TYPES = ("Card3D", "Cube3D", "Sphere3D", "ReadGeo3D")
OUTPUT_TYPES.update({kind: "geometry" for kind in GEOMETRY_TYPES})
OUTPUT_TYPES.update({"ReadSplat3D": "scene", "ReadAlembic3D": "scene", "ReadAlembicCamera3D": "camera", "ReadUSD3D": "scene", "ReadUSDCamera3D": "camera", "ReadGLTF3D": "scene", "Light3D": "light", "Camera3D": "camera", "Scene3D": "scene", "Project3D": "scene", "WriteGeo3D": "scene", "Render3D": "image", "Axis3D": "scene", "TransformGeo3D": "geometry"})
# A slot accepts a tuple of value types. Scene3D members may be geometry, lights or whole scenes
# (nesting is the hierarchy: a child scene inherits its parent's transform).
INPUT_TYPES = {"image": ("image",), "scene": ("scene",), "camera": ("camera",),
               "geometry": ("geometry", "scene"),
               # Axis3D's single slot accepts the same members a Scene3D object slot does.
               "object": ("geometry", "light", "scene"),
               # TransformGeo3D bakes vertices directly, so it takes one geometry, never a scene.
               "geo": ("geometry",)}
INPUT_TYPES.update({f"object{i}": ("geometry", "light", "scene") for i in range(8)})
INPUT_TYPES.update({f"light{i}": ("light",) for i in range(8)})
LIMITS = {"splat_relight": (0.0, 1.0), "splat_shadow_catch": (0.0, 1.0), "splat_sh_degree": (0, 3), "splat_opacity": (0.0, 1000000.0),
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
          "glow_threshold": (-10.0, 10.0), "glow_size": (0.0, 500.0), "brightness": (0.0, 100.0),
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
          # Shared fraction-of-the-box-radius softness for both Radial and Rectangle.
          "softness": (0.0, 1.0),
          "z_slice": (-100000.0, 100000.0), "octaves": (1, 8), "lacunarity": (0.01, 8.0),
          "seed": (0, 2147483647), "font_size": (1.0, 2000.0),
          # Keyer's four-point range ramp (group c3): unbounded like Clamp's minimum/maximum,
          # since a keyed quantity can legally sit outside 0..1 on HDR footage.
          "range_a": (-1000000.0, 1000000.0), "range_b": (-1000000.0, 1000000.0),
          "range_c": (-1000000.0, 1000000.0), "range_d": (-1000000.0, 1000000.0),
          # HueKeyer's simplified hue + saturation range.
          "hue_center": (0.0, 360.0), "hue_width": (0.0, 360.0), "hue_softness": (0.0, 180.0),
          "sat_min": (0.0, 1.0), "sat_max": (0.0, 1.0)}
LIMITS.update({"diffuse": (0.0, 1.0), "specular": (0.0, 1.0)})
LIMITS.update({name: (-1000000.0, 1000000.0) for name in
               ("tx", "ty", "tz", "rx", "ry", "rz", "roll", "target_x", "target_y", "target_z")})
LIMITS.update({"sx": (0.001, 1000.0), "sy": (0.001, 1000.0), "sz": (0.001, 1000.0),
               "card_width": (0.001, 100000.0), "card_height": (0.001, 100000.0),
               "cube_size": (0.001, 100000.0), "sphere_radius": (0.001, 100000.0),
               "segments": (3, 128), "intensity": (0.0, 1000.0), "ambient": (0.0, 10.0),
               "spec_amount": (0.0, 1.0), "spec_shininess": (1.0, 1024.0),
               "emission": (0.0, 1000.0), "samples": (1, 4),
               "fov": (1.0, 179.0), "near": (0.0001, 1000000.0), "far": (0.001, 1000000.0)})

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
                    "average", "from", "hypot")
TRANSFORM_FILTERS = ("nearest", "bilinear", "cubic")
TRACKER_MODES = ("match_move", "stabilise")
# Write destinations. Deliberately the two formats media.py can actually write; a format list
# longer than the writer is a promise the render button cannot keep.
WRITE_FILE_TYPES = ("Auto", "exr", "png")
EXR_BIT_DEPTHS = ("half", "float")
# Each ChannelShuffle output names its source explicitly. "0"/"1" are constants; there is no
# "leave it alone" option, because that is the one that hides a mistake.
CHANNEL_SOURCES = ("A.r", "A.g", "A.b", "A.a", "B.r", "B.g", "B.b", "B.a", "0", "1")
CHOICES = {"splat_orientation": ["as_authored", "colmap"],
           "splat_colorspace": ["srgb", "linear"],
           "rot_order": ["XYZ", "XZY", "YXZ", "YZX", "ZXY", "ZYX"], "colorspace": ["Auto", "sRGB", "Linear Rec.709", "ACEScg", "ACES2065-1", "Raw"],
           "alpha_mode": ["Auto", "Straight", "Premultiplied"],
           "operation": list(MERGE_OPERATIONS),
           "filter": list(TRANSFORM_FILTERS),
           "red_from": ["R", "G", "B", "A", "0", "1"], "green_from": ["R", "G", "B", "A", "0", "1"],
           "blue_from": ["R", "G", "B", "A", "0", "1"], "alpha_from": ["R", "G", "B", "A", "0", "1"],
           "missing": list(MISSING_FRAME_POLICIES),
           "justify": ["left", "center", "right"],
           "mode": list(TRACKER_MODES),
           "out_red": list(CHANNEL_SOURCES), "out_green": list(CHANNEL_SOURCES),
           "out_blue": list(CHANNEL_SOURCES), "out_alpha": list(CHANNEL_SOURCES),
           # Write output format. "Auto" reads the extension on the path rather than second-guessing
           # it, so renaming output.exr to output.png changes the writer and nothing else.
           "file_type": list(WRITE_FILE_TYPES), "bit_depth": list(EXR_BIT_DEPTHS),
           "project_outside": ["transparent", "clamp"], "project_backfaces": ["project", "skip"],
           "project_occlusion": ["off", "depth"],
           "shadows": ["off", "on"], "splat_cast_shadows": ["on", "off"],
           "render_backend": ["cpu", "auto", "gpu"],
           "render_mode": ["raster", "raytrace"],
           "light_type": ["Directional", "Point"], "render_output": ["rgba", "depth", "normals", "albedo", "diffuse",
                             "specular", "emission", "position", "uv", "object_id", "relight", "splats"],
           # Invert/Clamp/Multiply/Add/Gamma's channel selector. "rgba" also inverts/clamps alpha.
           "channels": ["rgb", "rgba", "alpha"],
           # Copy: which of A's channels replaces each of B's; "none" leaves that channel as B's own.
           "copy_red": ["none", "A.r", "A.g", "A.b", "A.a"], "copy_green": ["none", "A.r", "A.g", "A.b", "A.a"],
           "copy_blue": ["none", "A.r", "A.g", "A.b", "A.a"], "copy_alpha": ["none", "A.r", "A.g", "A.b", "A.a"],
           # ChannelMerge: single-channel source/destination selectors.
           "a_channel": ["A.r", "A.g", "A.b", "A.a"], "b_channel": ["B.r", "B.g", "B.b", "B.a"],
           "out_channel": ["R", "G", "B", "A"],
           # Keyer's keyed quantity (group c3).
           "keyer_operation": ["luminance", "red", "green", "blue", "saturation", "min", "max"]}


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
                if isinstance(node, dict) and node.get("type") == "Render3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("render_backend", "cpu")
                        params.setdefault("render_mode", "raster")
                if isinstance(node, dict) and node.get("type") == "ReadSplat3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("splat_relight", 0.0)
                        params.setdefault("splat_shadow_catch", 0.0)
                        params.setdefault("splat_cast_shadows", "on")
                if isinstance(node, dict) and node.get("type") == "Light3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("shadows", "off")
                if isinstance(node, dict) and node.get("type") == "Project3D":
                    params = node.get("params")
                    if isinstance(params, dict):
                        params.setdefault("project_occlusion", "off")
    return doc


def empty_document():
    return {"version": SCHEMA_VERSION, "nodes": {}, "view": None, "time": dict(DEFAULT_TIME),
            "animation": {"curves": {}}, "settings": copy.deepcopy(DEFAULT_SETTINGS),
            "node_data": {}, "expressions": {}, "references": []}


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {"color", "viewer"}:
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
    if not isinstance(viewer, dict) or set(viewer) != {"background"}:
        raise ValueError("settings.viewer is malformed")
    if viewer["background"] not in ("black", "checker"):
        raise ValueError("Viewer background must be black or checker")


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
                    "operations": ["describe", "inspect", "create", "set", "connect", "move", "rename", "label", "thumbnail", "disable", "delete", "reference", "view", "time", "settings", "set_key", "delete_key", "clear_curve", "set_shapes", "set_tracks", "set_expression", "clear_expression", "batch", "undo", "redo", "save", "load"]}
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
            nodes[key] = {"type": kind, "name": cmd.get("name", kind), "params": params,
                          "inputs": {slot: None for slot in
                                     list(SPECS[kind]["inputs"]) + list(SPECS[kind].get("optional_inputs", []))},
                          "pos": cmd.get("pos", [0, 0]), "disabled": False}
            return {"id": key}
        if op == "view":
            target = cmd.get("id")
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
            return {}
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
            if doc["view"] == key:
                doc["view"] = None
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
