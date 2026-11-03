"""Emboss and BumpBoss as the Nuke 17.0 Reference Guide describes them (lane 9, 2D behaviour step O).

Emboss: "Produces an emboss effect by offsetting the original input using the Angle and Width controls". Emboss Type chooses the output:
`traditional` "applies the emboss effect to the input image" (the relief is added to the picture), `effect` "outputs the effect only"
(the relief about mid grey). `edge_detector` is the guide's Edge Detector (simple, sobel, prewitt: how the slope of the brightness is read;
sobel by default, older documents `simple`). The guide gives no formula; the relief is the directional difference of the brightness along `angle`,
scaled by `width` (NodeBased's own reading, as before). `edge_channels` is the guide's edgechannels, "the edge detection is only applied to
these channels": the brightness the slope is read from is the mean of those channels (rgb by default, which is what the node always read; none
gives no relief). `blur_size` and `blur_quality` are the PreBlur tab's Blur size and Blur quality: the brightness is blurred before the slope is
read, with the blur EdgeDetect uses (a Gaussian; a wide one is worked on a scaled-down picture, `blur_quality` being the largest size worked at
full resolution). The guide's Threshold, Output and Erode size have no description there and are not built. Angle defaults to 0. An older document's node has no `emboss_type` and reads
`effect`, which is what it always drew, and keeps its own angle (`upgrade_params`).

BumpBoss: "an emboss effect by overlaying input A over input B ... modified by the offset between the center and lightposition
indicators". The guide gives no formula either, so this is the smallest reading that obeys every sentence it does state:

* the bump's slope is A's `height_channel` differenced across the picture; the light shades input B ("the image sequence to emboss with
  input A"; B is the third slot, after the mask). With B unconnected the picture is A itself, as the one-input node drew it. B is read
  on A's pixel grid, the result has A's window, and the node's mix dissolves toward B (the image being embossed). A saved node
  (`legacy_bumpboss`) draws its grey emboss from A alone and has no B;
* the light leans from `center` toward `lightposition` (Nuke's y runs up); the slope along that direction, times `bumpsize`, times the
  light `light_height`, is the relief: "zero is level with the image, producing no emboss", negative puts the light behind (the relief flips);
* `intensity` scales the simulated light; the picture is multiplied by `1 + intensity * relief`, and "Minimum Shadow Intensity ...
  increasing this value decreases the amount of shadow allowed" is the floor that factor cannot go under;
* `light_angle` (NodeBased's own knob) turns the light direction further, in degrees.

An older document's node has no `bumpsize` and keeps the one-input grey emboss (a mid grey lit from `light_angle`) it always drew.
"""
from __future__ import annotations

import math

import numpy as np

NEW = {
    "Emboss": {"emboss_type": "traditional", "edge_detector": "sobel", "edge_channels": "rgb", "blur_size": 0.0, "blur_quality": 15.0},
    "BumpBoss": {"lightposition_x": 480.0, "lightposition_y": 360.0, "bumpsize": 1.0, "light_height": 1.0, "intensity": 1.0,
                 "minshadow": 0.0, "center_x": 320.0, "center_y": 240.0, "legacy_bumpboss": 0},
}
UPGRADE = {
    "Emboss": {"emboss_type": "effect", "edge_detector": "simple", "edge_channels": "rgb", "blur_size": 0.0, "blur_quality": 15.0},
    "BumpBoss": {"lightposition_x": 480.0, "lightposition_y": 360.0, "bumpsize": 1.0, "light_height": 1.0, "intensity": 1.0,
                 "minshadow": 0.0, "center_x": 320.0, "center_y": 240.0, "legacy_bumpboss": 1},
}


def new_params(kind):
    return {**NEW.get(kind, {})}


def upgrade_params(kind):
    return dict(UPGRADE.get(kind, {}))


def _central(a, detector="simple"):
    """The slope of `a` along x and along the rows, per pixel: the central difference (`simple`), or that difference averaged over the
    three neighbouring rows (`prewitt`) or columns, the middle one counted twice (`sobel`)."""
    pad = np.pad(a, 1, mode="edge")
    dx = pad[1:-1, 2:] - pad[1:-1, :-2]
    dy = pad[2:, 1:-1] - pad[:-2, 1:-1]
    if detector in ("sobel", "prewitt"):
        up_x, down_x = pad[:-2, 2:] - pad[:-2, :-2], pad[2:, 2:] - pad[2:, :-2]
        left_y, right_y = pad[2:, :-2] - pad[:-2, :-2], pad[2:, 2:] - pad[:-2, 2:]
        mid = 2.0 if detector == "sobel" else 1.0
        scale = mid + 2.0
        dx = (up_x + mid * dx + down_x) / scale
        dy = (left_y + mid * dy + right_y) / scale
    return dx * 0.5, dy * 0.5


EDGE_CHANNELS = {"rgb": (0, 1, 2), "rgba": (0, 1, 2, 3), "all": (0, 1, 2, 3), "alpha": (3,), "red": (0,), "green": (1,), "blue": (2,), "none": ()}


def blur_reach(p):
    """The pixels each way the pre-blur reads (Blur size and Blur quality; the same blur EdgeDetect runs before its edges)."""
    from . import filterbehaviour
    return filterbehaviour.edge_blur_reach({"edge_blur_size": p.get("blur_size", 0.0), "edge_blur_quality": p.get("blur_quality", 15.0)})


def emboss(image, p, origin=(0, 0)):
    from . import filterbehaviour
    channels = EDGE_CHANNELS.get(str(p.get("edge_channels", "rgb")), (0, 1, 2))
    # The brightness the slope is read from: the mean of the Edge Channels, after the pre-blur. No channel, no slope, no relief.
    blurred = filterbehaviour.edge_pre_blur(image, {"edge_blur_size": p.get("blur_size", 0.0), "edge_blur_quality": p.get("blur_quality", 15.0)}, origin)
    lum = blurred[..., list(channels)].mean(axis=2) if channels else np.zeros(image.shape[:2], np.float32)
    dx, dy = _central(lum, str(p.get("edge_detector", "simple")))
    angle = math.radians(float(p.get("angle", 135.0)))
    relief = float(p.get("width", 1.0)) * (np.cos(angle) * dx + np.sin(angle) * dy)
    out = image.copy()
    if str(p.get("emboss_type", "effect")) == "effect":
        out[..., :3] = np.clip(0.5 + relief, 0, 1)[..., None]
    else:
        out[..., :3] = image[..., :3] + relief[..., None].astype(np.float32)
    return out


def light_direction(p):
    """The unit (x, row) direction the light leans from `center` toward `lightposition` (rows run down, Nuke's y up), turned by
    `light_angle`; (0, 0) when the two indicators coincide."""
    lx = float(p.get("lightposition_x", 480.0)) - float(p.get("center_x", 320.0))
    ly = -(float(p.get("lightposition_y", 360.0)) - float(p.get("center_y", 240.0)))
    norm = math.hypot(lx, ly)
    if norm < 1e-9:
        return 0.0, 0.0
    ux, uy = lx / norm, ly / norm
    turn = math.radians(float(p.get("light_angle", 0.0)))
    return ux * math.cos(turn) - uy * math.sin(turn), ux * math.sin(turn) + uy * math.cos(turn)


def bump_boss(image, p, height_index, b=None):
    """`image` is input A, the bump texture: its `height_channel` is differenced for the slope. `b` is input B, "the image sequence to
    emboss with input A" (same pixel grid as A); the light multiplies B's colour and B's alpha is kept. With no B the picture is A."""
    h = image[..., height_index]
    dx, dy = _central(h)
    ux, uy = light_direction(p)
    relief = float(p.get("bumpsize", 1.0)) * float(p.get("light_height", 1.0)) * (ux * dx + uy * dy)
    factor = np.maximum(float(p.get("minshadow", 0.0)), 1.0 + float(p.get("intensity", 1.0)) * relief)
    picture = image if b is None else b
    out = picture.copy()
    out[..., :3] = picture[..., :3] * factor[..., None].astype(np.float32)
    return out
