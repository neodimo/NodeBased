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

Toe's default lift changed from 0.05 to Nuke's 0 for new nodes: a node's params are stored whole, so no document changes and no knob is needed.
"""
from __future__ import annotations

SATURATION_MATH = ("Rec 709", "Ccir 601", "Average", "Maximum", "Rec 709 (old weights)")

NEW = {
    "ColorCorrect": {"divide_by_alpha": 0},
    "Saturation": {"luminance_math": "Rec 709"},
    "HueCorrect": {"older_curve_maths": 0},
    "HSVTool": {"adjust_by_scaling": 0},
    "SoftClip": {"older_grey": 0},
}
OLD = {
    "ColorCorrect": {"divide_by_alpha": 1},
    "Saturation": {"luminance_math": "Rec 709 (old weights)"},
    "HueCorrect": {"older_curve_maths": 1},
    "HSVTool": {"adjust_by_scaling": 1},
    "SoftClip": {"older_grey": 1},
}
LIMITS = {"divide_by_alpha": (0, 1), "older_curve_maths": (0, 1), "adjust_by_scaling": (0, 1), "older_grey": (0, 1)}
CHOICES = {"luminance_math": list(SATURATION_MATH)}


def new_params(kind):
    """What a new node of `kind` gets (Nuke's behaviour)."""
    return dict(NEW.get(kind, {}))


def upgrade_params(kind):
    """What an older document's node of `kind` gets, which keeps the pixels it had."""
    return dict(OLD.get(kind, {}))
