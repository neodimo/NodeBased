"""HSVTool's per-range mask knobs and its matte inversion (2D behaviour lane 2, step P).

Nuke's HSVTool lets each of the three ranges (hue, saturation, brightness) take its own mask channel (`HMask`, `SMask`, `BMask`) with an invert
checkbox, and inverts the matte it writes to the output channel (`invert_alpha_mask`). The defaults (no channel, nothing inverted) reproduce
what the node did before the knobs existed, so a saved graph renders the same pixels.

A mask socket wired to the node (`hue_mask`, `sat_mask`, `brt_mask`) keeps priority over the channel choice: the socket is the mask, and the
channel choice picks a channel of the input picture itself when no socket is wired.
"""
from __future__ import annotations

import numpy as np

CHANNELS = ["none", "rgba.red", "rgba.green", "rgba.blue", "rgba.alpha"]
# The three Reset buttons of the guide's Hue, Saturation and Brightness sections (`huereset`, `satreset`, `brtreset`; lane 2 step X5). Each button puts
# its section's range, adjustment and rolloff back to the defaults in one undo step; the counter beside it records the press, as Reload's does, so
# the press is a change the document and the undo history can hold. It changes no pixel by itself.
RESET_COUNTERS = {"huereset": 0, "satreset": 0, "brtreset": 0}
RESETS = {"huereset": ("hue_range_min", "hue_range_max", "hue_rotation", "hue_rolloff"),
          "satreset": ("saturation_range_min", "saturation_range_max", "sat_adjust", "saturation_rolloff"),
          "brtreset": ("brightness_range_min", "brightness_range_max", "brt_adjust", "brightness_rolloff")}


def reset_commands(key, params, counter, defaults):
    """The commands of one Reset button: each knob of its section back to `defaults`, and its counter up by one; run as one batch they are one undo step."""
    commands = [{"op": "set", "id": key, "param": name, "value": defaults[name]} for name in RESETS[counter]]
    commands.append({"op": "set", "id": key, "param": counter, "value": int(params.get(counter, 0)) + 1})
    return commands


NEW = {"HMask": "none", "invert_hue_mask": 0, "SMask": "none", "invert_sat_mask": 0, "BMask": "none", "invert_bright_mask": 0, "invert_alpha_mask": 0,
       **RESET_COUNTERS}
LIMITS = {"invert_hue_mask": (0, 1), "invert_sat_mask": (0, 1), "invert_bright_mask": (0, 1), "invert_alpha_mask": (0, 1),
          **{name: (0, 2147483647) for name in RESET_COUNTERS}}
CHOICES = {"HMask": CHANNELS, "SMask": CHANNELS, "BMask": CHANNELS}


def range_mask(image, p, socket, name, invert_name, socket_alpha):
    """One range's mask as a 2-D array, or None when it has none: the wired socket's alpha, else the picture's own channel `name` picks."""
    if socket is not None:
        mask = np.asarray(socket_alpha(socket), np.float64)
    else:
        choice = str(p.get(name, "none"))
        if choice not in CHANNELS or choice == "none":
            return None
        mask = np.asarray(image[..., CHANNELS.index(choice) - 1], np.float64)
    if int(p.get(invert_name, 0)):
        mask = 1.0 - mask
    return mask


# Vectorfield (same step): the file type, the reload counter and the two .3dl knobs that change the output.
VECTORFIELD = {"reload": 0, "vector_file_type": "auto", "vector_output_bitdepth": "auto", "vector_normalize_output": 0,
               # Lane 9 step X4: GPU extrapolate and 3DL file format. Both are held on the node as Nuke holds them; neither changes NodeBased's pixels
               # (the table is clamped at its edge on every path, and the .3dl reader takes the format from the file itself).
               "vector_gpu_extrapolate": 1, "vector_format_3dl": "auto"}
def upgrade_params(kind):
    """What a Vectorfield saved before each of its knobs existed gets on load: the knobs at the values that change nothing."""
    if kind == "HSVTool":
        return dict(RESET_COUNTERS)                    # the Reset buttons' counters (lane 2 step X5)
    return dict(VECTORFIELD) if kind == "Vectorfield" else {}


VECTORFIELD_LIMITS = {"vector_normalize_output": (0, 1), "vector_gpu_extrapolate": (0, 1)}
VECTORFIELD_CHOICES = {"vector_file_type": ["auto", "cube", "3dl"], "vector_output_bitdepth": ["auto", "8", "10", "12", "16"],
                       "vector_format_3dl": ["auto", "autodesk flame/smoke", "autodesk lustre", "assimilate scratch"]}
