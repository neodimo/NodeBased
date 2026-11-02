"""Knobs that switch a colour node between Nuke's behaviour and the behaviour NodeBased had before (2D behaviour lane 2).

A knob here is on in an older document's node and off in a new node, where the node does what Nuke's does (the same convention as
`gradeknobs.offset_by_alpha`). The older behaviour stays reachable, and a document saved before the knob existed loads with it on
(`ops2d_chrome.upgrade_nodes` sets it), so an old graph renders the same pixels. Each node's own kernel reads its knob; where
a node is run directly with a parameter dict that lacks the knob, the new node's value applies.

    ColorCorrect  divide_by_alpha     1: divide the colour by alpha before correcting and multiply after. Nuke works on the stored
                                      colour (its own (un)premult by does the dividing when asked).
    Saturation    luminance_math      "Rec 709 (old weights)": the 0.2126 / 0.7152 / 0.0722 weights the node always used. Nuke's
                                      Rec 709 is 0.2125 / 0.7154 / 0.0721.
    HueCorrect    older_curve_maths   1: the lum curve scales by saturation, the suppression curves pull a channel toward the largest
                                      of the other two (0 = nothing, 1 = fully), and the grey is 0.2126 / 0.7152 / 0.0722.
    HSVTool       adjust_by_scaling   1: saturation and brightness adjustments scale the value by (1 + adjustment). Nuke adds the
                                      adjustment to the value.
    SoftClip      older_grey          1: "preserve hue and brightness" moves the colour toward the 0.2126 / 0.7152 / 0.0722 grey.
                                      Nuke's Rec 709 is 0.2125 / 0.7154 / 0.0721.
    Colorspace    divide_by_alpha     1: divide the colour by alpha before converting and multiply it back after; Nuke converts the stored colour.
                  bradford_matrix     1: adapt the colour between two illuminants with the Bradford matrix, as NodeBased always did. Nuke's knob is
                                      off by default, which keeps the XYZ values equal and lets the RGB values change.
    ColorLookup   older_range_scaling 1: `range` scales the curves' input axis (values are divided by it first). Nuke's range only sets how far
                                      its precomputed table reaches, which changes how fast the lookup runs and not what it maps.
    Expression    older_expression_syntax
                                      1: the expressions are Python syntax (`a if c else b`, `and`, `**`), x and y are integer pixel
                                      indices counted from the top row, a blank expression is an error and a result that is not finite is
                                      refused. Nuke's are C-like (`c ? a : b`, `&&`), y counts up from the bottom, x and y are pixel centres
                                      (a half pixel in), a blank expression is zero and the arithmetic is plain 32-bit float.
    HistEQ        histeqbins          256: the histogram held 256 entries. Nuke's Histogram Entries default is 4096 (a real knob, so it is
                                      the value an older document holds rather than a switch).
    HueKeyer      older_hue_window    1: the matte is a hue window (Hue center, Hue width, Hue softness, a saturation range) and not the amount and
                                      saturation threshold curves over hue that Nuke's HueKeyer holds in its `lut` knob. Set on graphs saved before.
    ChannelMerge  older_channel_operations
                                      1: the operation is the older `operation` choice (NodeBased's Merge vocabulary, over by default, with the formulas it had), not
                                      the fourteen operations the guide lists for ChannelMerge (`channel_operation`, union by default). Set on graphs saved before.
    Log2Lin       older_log_maths     1: the file value is a code value on its own scale (no division by 1023), a pure power of
                                      (value - black) / (white - black) with gamma as the exponent.
    PLogLin       older_log_maths     1: the file value is a raw code value, density per code value alone sets the slope (no division
                                      by the negative gamma), alpha is left alone and the node only converts log to linear.

Toe's default lift changed from 0.05 to Nuke's 0 for new nodes: a node's params are stored whole, so no document changes and no knob is needed.
"""
from __future__ import annotations

# A curve over hue (degrees, 0 red, 120 green, 240 blue) that is 0 everywhere: Nuke's HueKeyer starts with no hue keyed and no saturation rejected.
FLAT_ZERO = '{"interpolation":"linear","points":[[0,0],[360,0]]}'
SATURATION_MATH = ("Rec 709", "Ccir 601", "Average", "Maximum", "Rec 709 (old weights)")

NEW = {
    "ColorCorrect": {"divide_by_alpha": 0},
    "Saturation": {"luminance_math": "Rec 709"},
    "HueCorrect": {"older_curve_maths": 0},
    "HSVTool": {"adjust_by_scaling": 0},
    "SoftClip": {"older_grey": 0},
    "Log2Lin": {"older_log_maths": 0},
    "PLogLin": {"older_log_maths": 0},
    "Colorspace": {"divide_by_alpha": 0, "bradford_matrix": 0},
    "ColorLookup": {"older_range_scaling": 0},
    "Expression": {"older_expression_syntax": 0},
    "HistEQ": {"histeqbins": 4096},
    "HueKeyer": {"older_hue_window": 0, "curve_amount": FLAT_ZERO, "curve_sat_thrsh": FLAT_ZERO},
    "ChannelMerge": {"channel_operation": "union", "older_channel_operations": 0},
}
OLD = {
    "ColorCorrect": {"divide_by_alpha": 1},
    "Saturation": {"luminance_math": "Rec 709 (old weights)"},
    "HueCorrect": {"older_curve_maths": 1},
    "HSVTool": {"adjust_by_scaling": 1},
    "SoftClip": {"older_grey": 1},
    "Log2Lin": {"older_log_maths": 1},
    "PLogLin": {"older_log_maths": 1},
    "Colorspace": {"divide_by_alpha": 1, "bradford_matrix": 1},
    "ColorLookup": {"older_range_scaling": 1},
    "Expression": {"older_expression_syntax": 1},
    "HistEQ": {"histeqbins": 256},
    "HueKeyer": {"older_hue_window": 1, "curve_amount": FLAT_ZERO, "curve_sat_thrsh": FLAT_ZERO},
    "ChannelMerge": {"channel_operation": "union", "older_channel_operations": 1},
}
LIMITS = {"divide_by_alpha": (0, 1), "older_curve_maths": (0, 1), "adjust_by_scaling": (0, 1), "older_grey": (0, 1), "older_log_maths": (0, 1), "bradford_matrix": (0, 1), "older_range_scaling": (0, 1), "older_expression_syntax": (0, 1), "histeqbins": (2, 65536), "older_hue_window": (0, 1), "older_channel_operations": (0, 1)}
# The operations the Reference Guide lists for ChannelMerge, in its order (the guide's names; "b if not a" is its `A?A:B`).
CHANNEL_MERGE_OPERATIONS = ("absminus", "b if not a", "divide", "from", "in", "max", "min", "minus", "multiply", "out", "plus", "stencil", "union", "xor")
CHOICES = {"luminance_math": list(SATURATION_MATH), "channel_operation": list(CHANNEL_MERGE_OPERATIONS)}


def new_params(kind):
    """What a new node of `kind` gets (Nuke's behaviour)."""
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What an older document's node of `kind` gets, which keeps the pixels it had."""
    return dict(OLD.get(kind, {}))
