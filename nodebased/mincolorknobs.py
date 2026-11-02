"""MinColor as Nuke has it (2D behaviour lane 2, step B): it adds a delta to the image so that the darkest value becomes the target.

Nuke's page: "Calculates the difference between the darkest value in the input image and the target value, and then adds the result to the image. This sets the
darkest pixel to the target value. To have MinColor calculate the darkest pixels and remember them, click Find Pixel Delta." So the node itself adds the delta it
remembers (`pixeldelta`) to the channels named by `modify`, when `direction` allows (Up: only a positive delta, Down: only a negative one, Both: either), and
Find Pixel Delta looks for the darkest value of the `search` channels over the frame range and stores `target` minus that value.

NodeBased's MinColor also finds the darkest or brightest pixel of a box and stores its colour (the Result knobs); that analysis is unchanged.
"""
from __future__ import annotations

import numpy as np

from .nukeexpr import channel_set


def applies(delta, direction):
    direction = str(direction).strip().lower()
    return (direction == "up" and delta > 0) or (direction == "down" and delta < 0) or (direction == "both" and delta != 0)


def add_delta(image, p):
    """`image` with the stored delta added to the `modify` channels, or the image itself when the direction says the delta stays out."""
    delta = float(p.get("pixeldelta", 0.0))
    if not applies(delta, p.get("adjdir", "Up")):
        return image
    out = image.copy()
    out[..., channel_set(p.get("modify", "all"), "all")] += np.float32(delta)
    return out


def search_minimum(pixels, data_origin, box, channels):
    """The smallest value of the searched channels inside the box (0-size box: the whole image), or None when the box holds no pixels."""
    crop = pixels
    x, y, w, h = box
    if w > 0 and h > 0:
        px, py = max(0, int(x - data_origin[0])), max(0, int(y - data_origin[1]))
        crop = pixels[py:py + int(h), px:px + int(w)]
    picked = channel_set(channels, "rgb")
    if crop.size == 0 or not picked:
        return None
    return float(crop[..., picked].min())


def find_pixel_delta(minima, target):
    """target - (the darkest value over the frames); `minima` holds one search_minimum per frame. None when there was nothing to search."""
    found = [m for m in minima if m is not None]
    return None if not found else float(target) - min(found)
