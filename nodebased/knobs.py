"""Presentation metadata for node properties (kept separate from the document schema)."""
from __future__ import annotations

from dataclasses import dataclass

from .core import CHOICES, LIMITS, SPECS


KNOB_KINDS = ("float_slider", "int", "bool", "enum", "xy", "xyz", "float", "color", "file_read",
              "file_write", "string", "multiline")


def _default_label(params):
    known_stems = {
        ("translate_x", "translate_y"): "Translate",
        ("center_x", "center_y"): "Center",
        ("red", "green", "blue", "alpha"): "Color",
    }
    if params in known_stems:
        return known_stems[params]
    return params[0].replace("_", " ").title()


@dataclass(frozen=True)
class KnobGroup:
    kind: str
    params: tuple[str, ...]
    label: str | None = None
    soft_range: tuple[float, float] | None = None

    def __post_init__(self):
        if self.label is None:
            object.__setattr__(self, "label", _default_label(self.params))


def _groups(*groups):
    return list(groups)


KNOB_LAYOUT = {
    "Relight": _groups(
        KnobGroup("color", ("red", "green", "blue"), label="Ambient"),
        KnobGroup("float_slider", ("diffuse",), label="Diffuse", soft_range=(0, 1)),
        KnobGroup("float_slider", ("specular",), label="Specular", soft_range=(0, 1)),
        KnobGroup("float_slider", ("mix",), label="Mix", soft_range=(0, 1))),
    "ReadSplat3D": _groups(
        KnobGroup("string", ("splat_path",), label="Splat file"),
        KnobGroup("enum", ("splat_orientation",)), KnobGroup("enum", ("splat_colorspace",)),
        KnobGroup("int", ("splat_sh_degree",)),
        KnobGroup("float_slider", ("splat_relight",), label="Relight", soft_range=(0, 1)),
        KnobGroup("float_slider", ("splat_shadow_catch",), label="Catch shadows", soft_range=(0, 1)),
        KnobGroup("enum", ("splat_cast_shadows",), label="Cast shadows"),
        KnobGroup("float", ("splat_opacity",), label="Opacity"),
        KnobGroup("float", ("splat_scale",), label="Splat scale"),
        KnobGroup("enum", ("rot_order",), label="Rotation order"),
        KnobGroup("xyz", ("tx", "ty", "tz"), label="Translate"),
        KnobGroup("xyz", ("rx", "ry", "rz"), label="Rotate"),
        KnobGroup("xyz", ("sx", "sy", "sz"), label="Scale"),
        KnobGroup("float", ("uscale",), label="Uniform scale"),
        KnobGroup("xyz", ("pivot_x", "pivot_y", "pivot_z"), label="Pivot")),
    "Read": _groups(
        KnobGroup("file_read", ("path",)), KnobGroup("enum", ("colorspace",)),
        KnobGroup("enum", ("alpha_mode",)), KnobGroup("string", ("layer",)),
        KnobGroup("int", ("subimage",)), KnobGroup("int", ("frame_offset",)),
        KnobGroup("enum", ("missing",))),
    "Constant": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("color", ("red", "green", "blue", "alpha"))),
    "Checker": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("int", ("size",))),
    "Grade": _groups(
        KnobGroup("float_slider", ("exposure",), soft_range=(-10, 10)),
        KnobGroup("float_slider", ("multiply",), soft_range=(-4, 4)),
        KnobGroup("float_slider", ("offset",), soft_range=(-10, 10)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "ColorCorrect": _groups(
        KnobGroup("float_slider", ("lift",), soft_range=(-2, 2)),
        KnobGroup("float_slider", ("gamma",), soft_range=(0.1, 4)),
        KnobGroup("float_slider", ("gain",), soft_range=LIMITS["gain"]),
        KnobGroup("float_slider", ("saturation",), soft_range=LIMITS["saturation"]),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Blur": _groups(
        KnobGroup("float_slider", ("radius",), soft_range=(0, 100)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Invert": _groups(
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Clamp": _groups(
        KnobGroup("float_slider", ("minimum",), soft_range=(-1, 1)),
        KnobGroup("float_slider", ("maximum",), soft_range=(0, 4)),
        KnobGroup("bool", ("clamp_min",)), KnobGroup("bool", ("clamp_max",)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Multiply": _groups(
        KnobGroup("float_slider", ("multiply",), soft_range=(-4, 4)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Add": _groups(
        KnobGroup("float_slider", ("offset",), soft_range=(-10, 10)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Gamma": _groups(
        KnobGroup("float_slider", ("gamma",), soft_range=(0.1, 4)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Saturation": _groups(
        KnobGroup("float_slider", ("saturation",), soft_range=LIMITS["saturation"]),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Erode": _groups(
        KnobGroup("float_slider", ("erode_size",), label="Size", soft_range=(-100, 100)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Dilate": _groups(
        KnobGroup("float_slider", ("dilate_size",), label="Size", soft_range=(-100, 100)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Median": _groups(
        KnobGroup("float_slider", ("median_size",), label="Size", soft_range=(0, 20)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Sharpen": _groups(
        KnobGroup("float_slider", ("sharpen_amount",), label="Amount", soft_range=(0, 2)),
        KnobGroup("float_slider", ("sharpen_size",), label="Size", soft_range=(0, 20)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Glow": _groups(
        KnobGroup("float_slider", ("glow_threshold",), label="Threshold", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("glow_size",), label="Size", soft_range=(0, 100)),
        KnobGroup("float_slider", ("brightness",), label="Brightness", soft_range=(0, 4)),
        KnobGroup("color", ("red", "green", "blue"), label="Tint"),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Soften": _groups(
        KnobGroup("float_slider", ("soften_size",), label="Size", soft_range=(0, 100)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Defocus": _groups(
        KnobGroup("float_slider", ("defocus",), label="Defocus", soft_range=(0, 100)),
        KnobGroup("float_slider", ("aspect",), label="Aspect", soft_range=(0.25, 4)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "DirBlur": _groups(
        KnobGroup("enum", ("blur_type",), label="Type"),
        KnobGroup("float_slider", ("angle",), label="Angle", soft_range=(-180, 180)),
        KnobGroup("float_slider", ("length",), label="Length", soft_range=(0, 200)),
        KnobGroup("xy", ("center_x", "center_y"), label="Center"),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "DropShadow": _groups(
        KnobGroup("float_slider", ("angle",), label="Angle", soft_range=(-180, 180)),
        KnobGroup("float_slider", ("distance",), label="Distance", soft_range=(0, 100)),
        KnobGroup("float_slider", ("shadow_size",), label="Size", soft_range=(0, 100)),
        KnobGroup("float_slider", ("opacity",), label="Opacity", soft_range=(0, 1)),
        KnobGroup("color", ("red", "green", "blue"), label="Color"),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Position": _groups(
        KnobGroup("int", ("translate_x",), label="Translate X"),
        KnobGroup("int", ("translate_y",), label="Translate Y")),
    "BlackOutside": _groups(),
    "AdjustBBox": _groups(
        KnobGroup("int", ("numpixels",), label="Add pixels"),
        KnobGroup("bool", ("clip_to_format",), label="Clip to format")),
    "Exposure": _groups(
        KnobGroup("enum", ("exposure_mode",), label="Adjust in"),
        KnobGroup("float_slider", ("blackpoint",), label="Black point", soft_range=(-1, 1)),
        KnobGroup("bool", ("gang",), label="Gang"),
        KnobGroup("float_slider", ("red",), label="Red", soft_range=(-5, 5)),
        KnobGroup("float_slider", ("green",), label="Green", soft_range=(-5, 5)),
        KnobGroup("float_slider", ("blue",), label="Blue", soft_range=(-5, 5)),
        KnobGroup("enum", ("channels",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "HueCorrect": _groups(
        KnobGroup("float_slider", ("sat_red",), label="Sat red", soft_range=(0, 2)),
        KnobGroup("float_slider", ("sat_yellow",), label="Sat yellow", soft_range=(0, 2)),
        KnobGroup("float_slider", ("sat_green",), label="Sat green", soft_range=(0, 2)),
        KnobGroup("float_slider", ("sat_cyan",), label="Sat cyan", soft_range=(0, 2)),
        KnobGroup("float_slider", ("sat_blue",), label="Sat blue", soft_range=(0, 2)),
        KnobGroup("float_slider", ("sat_magenta",), label="Sat magenta", soft_range=(0, 2)),
        KnobGroup("float_slider", ("lum_red",), label="Lum red", soft_range=(0, 2)),
        KnobGroup("float_slider", ("lum_yellow",), label="Lum yellow", soft_range=(0, 2)),
        KnobGroup("float_slider", ("lum_green",), label="Lum green", soft_range=(0, 2)),
        KnobGroup("float_slider", ("lum_cyan",), label="Lum cyan", soft_range=(0, 2)),
        KnobGroup("float_slider", ("lum_blue",), label="Lum blue", soft_range=(0, 2)),
        KnobGroup("float_slider", ("lum_magenta",), label="Lum magenta", soft_range=(0, 2)),
        KnobGroup("float_slider", ("hue_shift",), label="Hue shift", soft_range=(-180, 180)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "ColorMatrix": _groups(
        KnobGroup("float_slider", ("matrix_00",), label="Matrix 0,0", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_01",), label="Matrix 0,1", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_02",), label="Matrix 0,2", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_10",), label="Matrix 1,0", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_11",), label="Matrix 1,1", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_12",), label="Matrix 1,2", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_20",), label="Matrix 2,0", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_21",), label="Matrix 2,1", soft_range=(-2, 2)),
        KnobGroup("float_slider", ("matrix_22",), label="Matrix 2,2", soft_range=(-2, 2)),
        KnobGroup("bool", ("invert",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Mirror": _groups(
        KnobGroup("bool", ("flip_x",), label="Flip horizontal"),
        KnobGroup("bool", ("flip_y",), label="Flip vertical"),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Transform": _groups(
        KnobGroup("xy", ("translate_x", "translate_y")),
        KnobGroup("float_slider", ("rotate",), soft_range=(-180, 180)),
        KnobGroup("float_slider", ("scale",), soft_range=(0.1, 4.0)),
        KnobGroup("xy", ("center_x", "center_y")), KnobGroup("enum", ("filter",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Crop": _groups(
        KnobGroup("int", ("x",)), KnobGroup("int", ("y",)),
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Reformat": _groups(
        KnobGroup("enum", ("reformat_type",), label="Type"),
        KnobGroup("enum", ("format",)),
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("float", ("pixel_aspect",), label="Pixel aspect"),
        KnobGroup("float_slider", ("scale",), soft_range=(0.1, 4.0)),
        KnobGroup("enum", ("resize_type",), label="Resize"),
        KnobGroup("bool", ("center",)), KnobGroup("bool", ("flip",)),
        KnobGroup("bool", ("flop",)), KnobGroup("bool", ("turn",)),
        KnobGroup("enum", ("filter",)), KnobGroup("bool", ("preserve_bbox",), label="Preserve bounding box"),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "CornerPin": _groups(
        KnobGroup("xy", ("to1_x", "to1_y"), label="To 1"),
        KnobGroup("xy", ("to2_x", "to2_y"), label="To 2"),
        KnobGroup("xy", ("to3_x", "to3_y"), label="To 3"),
        KnobGroup("xy", ("to4_x", "to4_y"), label="To 4"),
        KnobGroup("xy", ("from1_x", "from1_y"), label="From 1"),
        KnobGroup("xy", ("from2_x", "from2_y"), label="From 2"),
        KnobGroup("xy", ("from3_x", "from3_y"), label="From 3"),
        KnobGroup("xy", ("from4_x", "from4_y"), label="From 4"),
        KnobGroup("enum", ("direction",)), KnobGroup("enum", ("filter",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Shuffle": _groups(
        KnobGroup("enum", ("red_from",)), KnobGroup("enum", ("green_from",)),
        KnobGroup("enum", ("blue_from",)), KnobGroup("enum", ("alpha_from",))),
    "ChannelShuffle": _groups(
        KnobGroup("enum", ("out_red",)), KnobGroup("enum", ("out_green",)),
        KnobGroup("enum", ("out_blue",)), KnobGroup("enum", ("out_alpha",))),
    "Roto": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("bool", ("invert",))),
    "Ramp": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("xy", ("p0_x", "p0_y"), label="P0"), KnobGroup("xy", ("p1_x", "p1_y"), label="P1"),
        KnobGroup("color", ("color0_red", "color0_green", "color0_blue", "color0_alpha"), label="Colour 0"),
        KnobGroup("color", ("color1_red", "color1_green", "color1_blue", "color1_alpha"), label="Colour 1"),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Radial": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("xy", ("box_x", "box_y"), label="Area"),
        KnobGroup("xy", ("box_width", "box_height"), label="Area size"),
        KnobGroup("float_slider", ("softness",), soft_range=(0, 1)),
        KnobGroup("color", ("red", "green", "blue", "alpha")),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Rectangle": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("xy", ("box_x", "box_y"), label="Area"),
        KnobGroup("xy", ("box_width", "box_height"), label="Area size"),
        KnobGroup("float_slider", ("softness",), soft_range=(0, 1)),
        KnobGroup("color", ("red", "green", "blue", "alpha")),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Noise": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("float_slider", ("size",), soft_range=(2, 512)),
        KnobGroup("float_slider", ("z_slice",), soft_range=(-10, 10)),
        KnobGroup("int", ("octaves",)), KnobGroup("float_slider", ("lacunarity",), soft_range=(1, 4)),
        KnobGroup("float_slider", ("gain",), soft_range=(0, 1)),
        KnobGroup("float_slider", ("gamma",), soft_range=(0.1, 4)),
        KnobGroup("int", ("seed",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Text": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("multiline", ("message",)), KnobGroup("string", ("font",)),
        KnobGroup("float_slider", ("font_size",), soft_range=(4, 300)),
        KnobGroup("xy", ("box_x", "box_y"), label="Box"),
        KnobGroup("xy", ("box_width", "box_height"), label="Box size"),
        KnobGroup("enum", ("justify",)),
        KnobGroup("color", ("red", "green", "blue", "alpha")),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Tracker": _groups(
        KnobGroup("int", ("reference_frame",)), KnobGroup("enum", ("mode",)),
        KnobGroup("bool", ("apply_translate",)), KnobGroup("bool", ("apply_rotate",)),
        KnobGroup("bool", ("apply_scale",)), KnobGroup("enum", ("filter",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Merge": _groups(
        KnobGroup("enum", ("operation",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Dissolve": _groups(
        KnobGroup("float_slider", ("which",), soft_range=(0, 1)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Keymix": _groups(
        KnobGroup("bool", ("invert_mask",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Copy": _groups(
        KnobGroup("enum", ("copy_red",)), KnobGroup("enum", ("copy_green",)),
        KnobGroup("enum", ("copy_blue",)), KnobGroup("enum", ("copy_alpha",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "ChannelMerge": _groups(
        KnobGroup("enum", ("a_channel",)), KnobGroup("enum", ("b_channel",)),
        KnobGroup("enum", ("operation",)), KnobGroup("enum", ("out_channel",)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Keyer": _groups(
        KnobGroup("enum", ("keyer_operation",), label="Operation"),
        KnobGroup("float_slider", ("range_a",), label="A", soft_range=(0, 1)),
        KnobGroup("float_slider", ("range_b",), label="B", soft_range=(0, 1)),
        KnobGroup("float_slider", ("range_c",), label="C", soft_range=(0, 1)),
        KnobGroup("float_slider", ("range_d",), label="D", soft_range=(0, 1)),
        KnobGroup("bool", ("invert",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "HueKeyer": _groups(
        KnobGroup("float_slider", ("hue_center",), label="Hue center", soft_range=(0, 360)),
        KnobGroup("float_slider", ("hue_width",), label="Hue width", soft_range=(0, 360)),
        KnobGroup("float_slider", ("hue_softness",), label="Hue softness", soft_range=(0, 180)),
        KnobGroup("float_slider", ("sat_min",), label="Saturation min", soft_range=(0, 1)),
        KnobGroup("float_slider", ("sat_max",), label="Saturation max", soft_range=(0, 1)),
        KnobGroup("bool", ("invert",)), KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "Difference": _groups(
        KnobGroup("float_slider", ("offset",), soft_range=(-1, 1)),
        KnobGroup("float_slider", ("gain",), soft_range=(0, 4)),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "TimeOffset": _groups(
        KnobGroup("int", ("time_offset",)), KnobGroup("bool", ("reverse",))),
    "FrameHold": _groups(
        KnobGroup("int", ("first_frame",)), KnobGroup("int", ("increment",))),
    "Retime": _groups(
        KnobGroup("int", ("input_range_start",), label="Input range start"),
        KnobGroup("int", ("input_range_end",), label="Input range end"),
        KnobGroup("int", ("output_range_start",), label="Output range start"),
        KnobGroup("int", ("output_range_end",), label="Output range end"),
        KnobGroup("float_slider", ("speed",), soft_range=(-4.0, 4.0))),
    "TimeClip": _groups(
        KnobGroup("enum", ("frame_range_type",), label="Frame range type"),
        KnobGroup("int", ("first",)), KnobGroup("int", ("last",)),
        KnobGroup("enum", ("before",)), KnobGroup("enum", ("after",)),
        KnobGroup("int", ("time_offset",))),
    "FrameRange": _groups(
        KnobGroup("int", ("first_frame",)), KnobGroup("int", ("last_frame",)),
        KnobGroup("enum", ("before",)), KnobGroup("enum", ("after",))),
    "AppendClip": _groups(
        KnobGroup("int", ("first_frame",)), KnobGroup("int", ("dissolve",)),
        *(KnobGroup("int", (f"length{i}",), label=f"Length {i}") for i in range(8))),
    "Grid": _groups(
        KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
        KnobGroup("xy", ("spacing_x", "spacing_y"), label="Spacing"),
        KnobGroup("int", ("number_x",), label="Lines in X"), KnobGroup("int", ("number_y",), label="Lines in Y"),
        KnobGroup("xy", ("grid_offset_x", "grid_offset_y"), label="Offset"),
        KnobGroup("float", ("line_width",)),
        KnobGroup("color", ("red", "green", "blue", "alpha")),
        KnobGroup("float_slider", ("mix",), soft_range=(0, 1))),
    "NoOp": _groups(KnobGroup("multiline", ("note",))),
    "Premult": [], "Unpremult": [], "Dot": [],
    "Switch": _groups(KnobGroup("int", ("which",))),
    "Viewer": [],
    "Write": _groups(
        KnobGroup("file_write", ("path",)), KnobGroup("enum", ("file_type",)),
        KnobGroup("enum", ("bit_depth",))),
}

# 3D panels follow Nuke: a vector is one row of typed fields, a size or a distance is a typed
# field, and a slider appears only where the value is a bounded scalar that is natural to scrub
# (an angle, a 0..1 amount, an intensity).
# Every node that carries a transform shows the same block in Nuke's order: rotation order,
# translate, rotate, scale, uniform scale, pivot.
_TRANSLATE_KNOB = KnobGroup("xyz", ("tx", "ty", "tz"), label="Translate")
_XFORM_KNOBS = (KnobGroup("enum", ("rot_order",), label="Rotation order"),
                _TRANSLATE_KNOB,
                KnobGroup("xyz", ("rx", "ry", "rz"), label="Rotate"),
                KnobGroup("xyz", ("sx", "sy", "sz"), label="Scale"),
                KnobGroup("float", ("uscale",), label="Uniform scale"),
                KnobGroup("xyz", ("pivot_x", "pivot_y", "pivot_z"), label="Pivot"))
_FORCE_KNOBS = (KnobGroup("float", ("probability",), label="Probability"),
                KnobGroup("int", ("from_frame",), label="From frame"),
                KnobGroup("int", ("to_frame",), label="To frame"),
                KnobGroup("int", ("seed",), label="Random seed"))
_SURFACE_KNOB = KnobGroup("color", ("red", "green", "blue", "alpha"))
# Specular amount is a 0..1 mix and scrubs well. Shininess and emission are open-ended magnitudes:
# a slider across their whole legal range put all the useful values in its first few pixels.
_MATERIAL_KNOBS = (KnobGroup("float_slider", ("spec_amount",), label="Specular", soft_range=LIMITS["spec_amount"]),
                   KnobGroup("float", ("spec_shininess",), label="Shininess"),
                   KnobGroup("float", ("emission",), label="Emission"))
_TARGET_KNOB = KnobGroup("xyz", ("target_x", "target_y", "target_z"), label="Look at")
# Nuke's own knob names for polygon amount, shared by Card3D, Sphere3D and Cylinder3D.
_ROWS_COLUMNS_KNOBS = (KnobGroup("int", ("rows",)), KnobGroup("int", ("columns",)))
KNOB_LAYOUT.update({
    "Card3D": _groups(KnobGroup("float", ("card_width",), label="Width"),
                      KnobGroup("float", ("card_height",), label="Height"),
                      *_ROWS_COLUMNS_KNOBS, *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Cube3D": _groups(KnobGroup("float", ("cube_size",), label="Size"),
                      *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Sphere3D": _groups(KnobGroup("float", ("sphere_radius",), label="Radius"),
                        KnobGroup("int", ("segments",), label="Segments (legacy)"),
                        *_ROWS_COLUMNS_KNOBS, *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Cylinder3D": _groups(KnobGroup("float", ("cyl_radius",), label="Radius"),
                          KnobGroup("float", ("cyl_height",), label="Height"),
                          *_ROWS_COLUMNS_KNOBS, KnobGroup("enum", ("cyl_caps",), label="Caps"),
                          *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "ReadAlembic3D": _groups(KnobGroup("string", ("abc_path",), label="Alembic file"),
                             KnobGroup("string", ("abc_root",), label="Root object")),
    "ReadAlembicCamera3D": _groups(KnobGroup("string", ("abc_path",), label="Alembic file"),
                                   KnobGroup("string", ("abc_camera",), label="Camera object")),
    "ReadUSD3D": _groups(KnobGroup("string", ("usd_path",), label="USD file"),
                         KnobGroup("string", ("usd_root",), label="Root prim")),
    "ReadUSDCamera3D": _groups(KnobGroup("string", ("usd_path",), label="USD file"),
                               KnobGroup("string", ("usd_camera",), label="Camera prim")),
    "ReadGLTF3D": _groups(KnobGroup("string", ("gltf_path",), label="glTF file"),
                          KnobGroup("string", ("gltf_root",), label="Root node")),
    "ReadGeo3D": _groups(KnobGroup("string", ("geo_path",), label="OBJ file"), *_XFORM_KNOBS, _SURFACE_KNOB, *_MATERIAL_KNOBS),
    "Light3D": _groups(KnobGroup("enum", ("light_type",), label="Type"),
                       KnobGroup("enum", ("shadows",), label="Shadows"),
                       KnobGroup("float", ("shadow_bias",), label="Shadow bias"),
                       KnobGroup("float", ("shadow_blur",), label="Shadow blur"),
                       KnobGroup("int", ("shadow_samples",), label="Shadow samples"),
                       _TRANSLATE_KNOB, _TARGET_KNOB,
                       KnobGroup("enum", ("falloff_type",), label="Falloff"),
                       KnobGroup("float", ("cone_angle",), label="Cone angle"),
                       KnobGroup("float", ("cone_penumbra_angle",), label="Cone penumbra angle"),
                       KnobGroup("float", ("cone_falloff",), label="Cone falloff"),
                       KnobGroup("color", ("red", "green", "blue"), label="Color"),
                       KnobGroup("float_slider", ("intensity",), soft_range=(0, 5))),
    "Camera3D": _groups(_TRANSLATE_KNOB, _TARGET_KNOB,
                        KnobGroup("float_slider", ("roll",), soft_range=(-180, 180)),
                        KnobGroup("float", ("focal",), label="Focal length (mm)"),
                        KnobGroup("float", ("haperture",), label="Horizontal aperture (mm)"),
                        KnobGroup("float", ("vaperture",), label="Vertical aperture (mm)"),
                        KnobGroup("float", ("near",)), KnobGroup("float", ("far",))),
    "Project3D": _groups(KnobGroup("enum", ("project_outside",), label="Outside"),
                         KnobGroup("enum", ("project_backfaces",), label="Backfaces"),
                         KnobGroup("enum", ("project_occlusion",), label="Occlusion")),
    "WriteGeo3D": _groups(KnobGroup("string", ("geo_write_path",), label="OBJ / USD file")),
    "Scene3D": _groups(*_XFORM_KNOBS),
    "Axis3D": _groups(*_XFORM_KNOBS),
    "TransformGeo3D": _groups(*_XFORM_KNOBS),
    "MergeGeo3D": _groups(*_XFORM_KNOBS),
    "Normals3D": _groups(KnobGroup("enum", ("normals_mode",), label="Mode"),
                         KnobGroup("bool", ("flip_winding",), label="Flip winding")),
    "DisplaceGeo3D": _groups(KnobGroup("float", ("displace_scale",), label="Scale"),
                             KnobGroup("float", ("displace_offset",), label="Offset"),
                             KnobGroup("enum", ("displace_channel",), label="Channel"),
                             KnobGroup("bool", ("recompute_normals",), label="Recompute normals")),
    "ParticleEmitter3D": _groups(
        KnobGroup("enum", ("emit_from",), label="Emit from"),
        KnobGroup("float", ("emit_rate",), label="Emission rate"),
        KnobGroup("enum", ("emit_rate_unit",), label="Rate unit"),
        KnobGroup("int", ("start_frame",), label="Start at"),
        KnobGroup("float", ("life",), label="Lifetime (frames)"),
        KnobGroup("float", ("life_variance",), label="Lifetime variation"),
        KnobGroup("float", ("emit_speed",), label="Speed"),
        KnobGroup("float", ("speed_variance",), label="Speed variation"),
        KnobGroup("xyz", ("emit_dir_x", "emit_dir_y", "emit_dir_z"), label="Direction"),
        KnobGroup("bool", ("direction_from_normals",), label="Direction from normals"),
        KnobGroup("float", ("spread",), label="Spread (degrees)"),
        KnobGroup("float", ("particle_size",), label="Size"),
        KnobGroup("float", ("size_variance",), label="Size variation"),
        KnobGroup("color", ("red", "green", "blue", "alpha"), label="Color"),
        KnobGroup("int", ("seed",), label="Random seed"),
        KnobGroup("int", ("substeps",), label="Substeps"),
        KnobGroup("int", ("max_particles",), label="Maximum particles"),
        *_XFORM_KNOBS),
    "ParticleGravity3D": _groups(
        KnobGroup("xyz", ("gravity_x", "gravity_y", "gravity_z"), label="Direction"),
        KnobGroup("float", ("strength",), label="Strength"), *_FORCE_KNOBS),
    "ParticleDrag3D": _groups(
        KnobGroup("float", ("drag",), label="Drag"),
        KnobGroup("float", ("drag_quadratic",), label="Quadratic drag"), *_FORCE_KNOBS),
    "ParticleWind3D": _groups(
        KnobGroup("xyz", ("wind_x", "wind_y", "wind_z"), label="Direction"),
        KnobGroup("float", ("strength",), label="Strength"),
        KnobGroup("float", ("wind_gust",), label="Gust"),
        KnobGroup("float", ("wind_gust_rate",), label="Gust rate (cycles per frame)"), *_FORCE_KNOBS),
    "ParticleTurbulence3D": _groups(
        KnobGroup("enum", ("turb_mode",), label="Field"),
        KnobGroup("float", ("turb_size",), label="Size"),
        KnobGroup("float", ("strength",), label="Strength"),
        KnobGroup("int", ("octaves",), label="Octaves"), *_FORCE_KNOBS),
    "ParticleBounce3D": _groups(
        KnobGroup("float", ("bounce",), label="Bounce"),
        KnobGroup("float", ("friction",), label="Friction"),
        KnobGroup("bool", ("kill_on_collision",), label="Kill on collision"), *_FORCE_KNOBS),
    "ParticleRender3D": _groups(
        KnobGroup("enum", ("representation",), label="Representation"),
        KnobGroup("float", ("size_scale",), label="Size scale")),
    "ParticleCache3D": _groups(KnobGroup("int", ("cache_memory_mb",), label="Memory budget (MB)"),
                               KnobGroup("int", ("cache_disk_mb",), label="Disk budget (MB)")),
    "Render3D": _groups(KnobGroup("int", ("width",)), KnobGroup("int", ("height",)),
                        KnobGroup("color", ("red", "green", "blue", "alpha"), label="Background"),
                        KnobGroup("float_slider", ("ambient",), soft_range=(0, 1)),
                        KnobGroup("int", ("samples",), label="Antialiasing samples"),
                        KnobGroup("enum", ("render_output",), label="Output"),
                        KnobGroup("enum", ("render_backend",), label="Backend"),
                        KnobGroup("enum", ("render_mode",), label="Mode")),
})


def _inverted_binary_name(param):
    return param in {"invert", "apply_translate", "apply_rotate", "apply_scale"}


def _scalar_kind(node_type, param, value=None):
    if param in CHOICES:
        return "enum"
    if param == "path" and node_type == "Read":
        return "file_read"
    if param == "path" and node_type == "Write":
        return "file_write"
    if isinstance(value, str):
        return "string"
    if _inverted_binary_name(param):
        return "bool"
    if param in {"which", "subimage"}:
        return "int"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "bool" if LIMITS.get(param) == (0, 1) else "int"
    return "float_slider"


def _fallback_layout(node_type):
    return [KnobGroup(_scalar_kind(node_type, param, default), (param,))
            for param, default in SPECS[node_type]["params"].items()]


def knob_layout(node_type):
    """Return presentation groups, deriving simple groups for newly added node types."""
    return KNOB_LAYOUT.get(node_type, _fallback_layout(node_type))


def resolve_kind(node_type, param, value):
    """Resolve one parameter to the kind used by its node's presentation layout."""
    for group in knob_layout(node_type):
        if param in group.params:
            return group.kind
    return _scalar_kind(node_type, param, value)
