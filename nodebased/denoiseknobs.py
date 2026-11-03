"""Denoise as the Nuke 17.0 Reference Guide describes its main tab (lane 9, 2D behaviour step L): Source, Noise Model, Profile, Output,
Lift Blacks, Roll Off, Smoothness, Luminance Blend, Preserve Edges and the `Noise` input.

What the guide states, and what this module does with it:

* Denoise Amount: the overall amount of noise removed; 0 removes none. Here it sets the colour threshold of an edge-preserving
  window average (the older node's rule, `max(0.01, amount)` and a window of `2 + 5 amount` pixels).
* Roll Off: "A value of 1 equals hard thresholding. Any other value produces soft thresholding between the Denoise Amount and the
  Roll Off multiplied by Denoise Amount." A pixel in the window counts fully when its colour is within the threshold T of the centre
  pixel, not at all beyond `rolloff * T`, and falls off linearly between (a hard step when rolloff is 1).
* Smoothness: "the smoothness of the denoised image. This affects the underlying image rather than the noise detected": it scales the
  window, so 0 averages nothing and 1 is the default window.
* Luminance Blend: "blends the denoised luminance with the image's original luminance, bringing back some of the image detail": 1 keeps
  the denoised luminance, 0 the original (Rec. 709 luminance; the denoised colour difference stays).
* Preserve Edges: "attempts to sharpen the image at edges to prevent over-smoothing": an unsharp step on the denoised picture (a 3 x 3
  average subtracted at half strength; the guide gives no amount, so the half is ours).
* Lift Blacks: sub-blacks (values below 0) are lifted to 0 in the result.
* Output: Result is the denoised image, Noise "the noise that was removed from the Source image": source minus result, so that
  source - noise is the result exactly.
* Source (Film or Digital) and Noise Model (Constant or Modulated): Constant removes the same amount across all intensities; Modulated
  scales the threshold with brightness. The guide's text says digital footage carries its noise in the light regions and film in the
  dark ones, but gives no curve, so Modulated scales the threshold by 0.25 + 1.5 L (Digital) or 0.25 + 1.5 (1 - L) (Film), L the clipped
  luminance: the direction is the guide's, the slope is ours.
* Profile: Constant uses one noise level for the whole picture. With a `Noise` input wired that level is measured on that picture; with
  none it is the nominal level (1/3, so the threshold is the amount, as before). Automatic measures the level per luminance band on the
  Source itself ("Denoise always bases the noise profile on your Source footage, even if you've attached another clip to the Noise
  input"). The level is the robust (median absolute deviation) estimate of the luminance minus its 3 x 3 average, and the threshold is
  3 times the level times the amount.

Foundry's trained model, the analysis region box, the profile tuning curve, the frequency and channel gains and the `Motion` input have no
equivalent here; they are listed in PARITY_2D_FIDELITY.md.

An older document's node keeps its own reading (`legacy_denoise` 1): the Gaussian bilateral of `bilateralknobs`, no luminance blend, no
output choice. A call that carries no flag reads as the older node.
"""
from __future__ import annotations

import math

import numpy as np

TYPES = ("Digital", "Film")
MODELS = ("Constant", "Modulated")
PROFILES = ("Constant", "Automatic")
OUTPUTS = ("Result", "Noise")
NOMINAL_LEVEL = 1.0 / 3.0
MAX_RADIUS = 24
BANDS = 8
NEW = {"legacy_denoise": 0, "denoise_type": "Digital", "denoise_model": "Modulated", "denoise_profile": "Constant",
       "denoise_output": "Result", "denoise_lift_blacks": 0, "denoise_rolloff": 2.0, "denoise_smoothness": 1.0, "denoise_lumablend": 0.7, "denoise_preserve_edges": 0}
# What a saved node gets on load: the flag and every new knob at the value that leaves it as it was (the luminance blend 1 keeps the
# denoised luminance, the model Constant the same threshold everywhere).
UPGRADE = {"legacy_denoise": 1, "denoise_type": "Digital", "denoise_model": "Constant", "denoise_profile": "Constant",
           "denoise_output": "Result", "denoise_lift_blacks": 0, "denoise_rolloff": 2.0, "denoise_smoothness": 1.0, "denoise_lumablend": 1.0, "denoise_preserve_edges": 0}


def is_legacy(p):
    return bool(int(p.get("legacy_denoise", 1)))


def radius(p):
    """The window's reach in pixels each way (what a tile must read past its edge)."""
    strength = max(0.0, float(p.get("denoise_strength", 0.2)))
    base = 2.0 + 5.0 * strength
    if is_legacy(p):
        return int(math.ceil(base))
    reach = int(math.ceil(base * max(0.0, float(p.get("denoise_smoothness", 1.0)))))
    return min(MAX_RADIUS, reach) + (1 if int(p.get("denoise_preserve_edges", 0)) else 0)


def needs_full_frame(p, has_noise_input=False):
    """The noise level is measured on a whole picture, so a node that measures it cannot be cut into tiles."""
    return not is_legacy(p) and (has_noise_input or str(p.get("denoise_profile", "Constant")) == "Automatic")


def luminance(image):
    return (0.2126 * image[..., 0] + 0.7152 * image[..., 1] + 0.0722 * image[..., 2]).astype(np.float64)


def _box3(plane):
    padded = np.pad(plane, 1, mode="edge")
    h, w = plane.shape
    return sum(padded[dy:dy + h, dx:dx + w] for dy in range(3) for dx in range(3)) / 9.0


def _residual(plane):
    return plane - _box3(plane)


def _mad_level(residual):
    if residual.size == 0:
        return NOMINAL_LEVEL
    centre = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - centre)) / math.sqrt(8.0 / 9.0))


def noise_level(image):
    """The robust noise level of a picture: the MAD of its luminance minus the luminance's 3 x 3 average."""
    return _mad_level(_residual(luminance(image)))


def noise_profile(image):
    """The level per luminance band of `image`: (band centres, levels). A band with fewer than 16 pixels takes the whole picture's level."""
    lum = luminance(image)
    resid = _residual(lum)
    overall = _mad_level(resid.ravel())
    edges = np.linspace(0.0, 1.0, BANDS + 1)
    clipped = np.clip(lum, 0.0, 1.0)
    index = np.minimum((clipped * BANDS).astype(int), BANDS - 1)
    levels = np.empty(BANDS)
    for band in range(BANDS):
        chosen = resid[index == band]
        levels[band] = _mad_level(chosen) if chosen.size >= 16 else overall
    return 0.5 * (edges[:-1] + edges[1:]), levels


def threshold_map(image, p, noise=None):
    """The colour threshold T at each pixel: 3 x noise level x amount, scaled by the model's brightness curve."""
    amount = max(0.0, float(p.get("denoise_strength", 0.2)))
    lum = np.clip(luminance(image), 0.0, 1.0)
    profile = str(p.get("denoise_profile", "Constant"))
    if profile == "Automatic":
        centres, levels = noise_profile(image)
        level = np.interp(lum, centres, levels)
    else:
        level = NOMINAL_LEVEL if noise is None else max(1e-4, noise_level(noise))
        level = np.full(lum.shape, level)
        if str(p.get("denoise_model", "Modulated")) == "Modulated":
            bright = 1.0 - lum if str(p.get("denoise_type", "Digital")) == "Film" else lum
            level = level * (0.25 + 1.5 * bright)
    return np.maximum(0.01, 3.0 * amount * level)


def _window_average(image, p, thresh):
    rad = radius(p) - (1 if int(p.get("denoise_preserve_edges", 0)) else 0)
    if rad <= 0:
        return image.astype(np.float64)
    h, w = image.shape[:2]
    rolloff = max(1.0, float(p.get("denoise_rolloff", 2.0)))
    pad = np.pad(image.astype(np.float64), ((rad, rad), (rad, rad), (0, 0)), mode="edge")
    centre = image.astype(np.float64)
    out = np.zeros(image.shape, np.float64)
    total = np.zeros((h, w, 1), np.float64)
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            spatial = math.exp(-(dx * dx + dy * dy) / max(1.0, 2.0 * rad * rad))
            sample = pad[rad + dy:rad + dy + h, rad + dx:rad + dx + w]
            diff = sample[..., :3] - centre[..., :3]
            dist = np.sqrt(np.sum(diff * diff, axis=2))
            if rolloff == 1.0:
                colour = (dist <= thresh).astype(np.float64)
            else:
                colour = np.clip((rolloff * thresh - dist) / ((rolloff - 1.0) * thresh), 0.0, 1.0)
            weight = (spatial * colour)[..., None]
            out += sample * weight
            total += weight
    return out / np.maximum(total, 1e-20)


def denoise(image, p, noise=None):
    """Denoise on an (h, w, 4) picture; `noise` (any shape) is the picture the noise level is measured on, None for the nominal level."""
    if is_legacy(p):
        from . import bilateralknobs
        strength = float(p.get("denoise_strength", 0.2))
        return bilateralknobs.bilateral(image, {"spatial_size": 2.0 + 5.0 * strength, "colour_sigma": max(0.01, strength)})
    if float(p.get("denoise_strength", 0.2)) <= 0.0:
        result = image.astype(np.float64)        # "A value of 0 removes no noise."
    else:
        result = _window_average(image, p, threshold_map(image, p, noise))
    if int(p.get("denoise_preserve_edges", 0)):
        result = result + 0.5 * (result - np.stack([_box3(result[..., c]) for c in range(result.shape[2])], axis=-1))
    blend = float(p.get("denoise_lumablend", 0.7))
    if blend != 1.0:
        shift = (1.0 - blend) * (luminance(image) - luminance(result))
        result[..., :3] += shift[..., None]
    if int(p.get("denoise_lift_blacks", 0)):
        result[..., :3] = np.maximum(result[..., :3], 0.0)
    result = result.astype(np.float32)
    if str(p.get("denoise_output", "Result")) == "Noise":
        return (image.astype(np.float32) - result).astype(np.float32)
    return result


LIMITS = {"denoise_rolloff": (1.0, 10.0), "denoise_smoothness": (0.0, 4.0), "denoise_lumablend": (0.0, 1.0),
          "denoise_lift_blacks": (0, 1), "denoise_preserve_edges": (0, 1), "legacy_denoise": (0, 1)}
CHOICES = {"denoise_type": list(TYPES), "denoise_model": list(MODELS), "denoise_profile": list(PROFILES), "denoise_output": list(OUTPUTS)}
