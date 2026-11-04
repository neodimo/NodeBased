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
NEW = {"HMask": "none", "invert_hue_mask": 0, "SMask": "none", "invert_sat_mask": 0, "BMask": "none", "invert_bright_mask": 0, "invert_alpha_mask": 0}
LIMITS = {"invert_hue_mask": (0, 1), "invert_sat_mask": (0, 1), "invert_bright_mask": (0, 1), "invert_alpha_mask": (0, 1)}
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
VECTORFIELD = {"reload": 0, "vector_file_type": "auto", "vector_output_bitdepth": "auto", "vector_normalize_output": 0}
VECTORFIELD_LIMITS = {"vector_normalize_output": (0, 1)}
VECTORFIELD_CHOICES = {"vector_file_type": ["auto", "cube", "3dl"], "vector_output_bitdepth": ["auto", "8", "10", "12", "16"]}
