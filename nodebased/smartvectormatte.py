"""SmartVector's optional foreground matte: Matte Channel picks where the mask comes from, Output picks which side's vectors leave.

Nuke's guide (SmartVector, Matte input and Matte Channel): the matte is "an optional matte of the foreground, which may improve
the motion estimation by reducing the dragging of pixels that can occur between foreground and background objects", and "when
Matte Channel is set to anything other than None, the Output control determines whether Foreground or Background vectors are
passed down stream or written to file".

The vectors are estimated on the frames with the other side blacked out, so the side that is kept is not dragged by the one that
is not, and the vectors of the side that is not output are absent: zero, with coverage (the layer's alpha) 0. The cut is at a
matte value of one half. `output` is Foreground or Background; a node with Matte Channel None estimates on the whole frame as before.
"""
import numpy as np

CHANNELS = ("None", "Source Alpha", "Source Inverted Alpha", "Matte Luminance", "Matte Inverted Luminance",
            "Matte Alpha", "Matte Inverted Alpha")
OUTPUTS = ("Background", "Foreground")
_LUMA = np.array([.2126, .7152, .0722], np.float32)


def needs_matte_input(channel):
    return channel.startswith("Matte")


def foreground_mask(channel, source, matte):
    """The foreground mask (height x width, 0 to 1) a Matte Channel names, from the Source and Matte pixels; None for "None"."""
    if channel == "None":
        return None
    if channel not in CHANNELS:
        raise ValueError(f"SmartVector: unknown matte channel {channel!r}")
    inverted = "Inverted" in channel
    if channel.startswith("Source"):
        value = np.asarray(source)[..., 3]
    elif channel.endswith("Luminance"):
        value = np.asarray(matte)[..., :3] @ _LUMA
    else:
        value = np.asarray(matte)[..., 3]
    value = np.clip(value, 0.0, 1.0).astype(np.float32)
    return 1.0 - value if inverted else value


def kept(mask, output):
    """1 where the output side is, 0 elsewhere (a hard cut at one half)."""
    if output not in OUTPUTS:
        raise ValueError(f"SmartVector: output must be one of {list(OUTPUTS)}")
    foreground = mask >= 0.5
    return (foreground if output == "Foreground" else ~foreground).astype(np.float32)


def apply(frames, masks, output):
    """The frames with everything off the output side set to zero, for the motion estimate."""
    return [f * kept(m, output)[..., None] for f, m in zip(frames, masks)]
