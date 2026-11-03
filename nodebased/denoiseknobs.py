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

Finish pass (lane 9, step L finish), the rest of the guide's tabs:

* Tune Channels: "Luminance Gain / Chrominance Gain: scales the denoising threshold for the luminance / chrominance channel". The window
  average runs once per channel group, with the threshold times that gain, and the result takes the luminance of the first and the colour
  difference (the picture minus its luminance) of the second. A gain of 0 removes none of that group's noise. Defaults 1 and 5.
* Tune Frequencies: "High / Medium / Low / Very Low Gain: scales the threshold for fine (...) noise removal ... A value of 0 means no fine
  noise is removed, whereas 1 is the estimated threshold" and each band can be left unprocessed. The noise is split into four bands with
  Gaussian blurs of 1, 2 and 4 pixels (high = the noise minus its 1 pixel blur, medium = the 1 minus the 2 pixel blur, low = the 2 minus
  the 4, very low = the 4 pixel blur itself). Each processed band is taken from a window average run with the threshold times that band's
  gain, and the bands are put back together to be taken off the picture. With every band on at gain 1 that is the whole noise, exactly.
  The cut-offs are ours.
* Profile tab: "Tune Profile" turns on Low / Mid / High Gain, which scale the threshold in the shadows, midtones and highlights (a line
  through gains at luminance 0, 0.5 and 1); the curve editor is the same idea drawn by hand: "the x axis represents image intensity ... the
  y axis the relative amount of noise removed", read as a multiplier on the threshold. Profile Frame is the frame the Automatic profile is
  measured on (the guide: "based on the entire Profile Frame"); the Recalculate Profile button has nothing to recalculate, the profile
  being measured live.
* Noise Analysis: the Analysis Region box (x, y, r, t in pixels, y from the bottom edge, as the guide lists them; all four 0 is no box:
  the whole picture, or the nominal level when no Noise input is wired) is where the Constant profile measures the noise, on the Noise
  input when one is wired and the Source otherwise, at the Analysis Frame. "The minimum size for the analysis region is 80x80 pixels. If
  the analysis region is too small, Denoise doesn't analyze the footage or remove any noise": a box under 80 pixels either way removes
  nothing. Lock Noise Analysis "performs no analysis": the node reads the level and profile it holds (set by Analyze Noise or Import).
  Analyze Noise measures them now, Export writes them to the Analysis File and Import reads them from it (and locks the analysis, as the
  guide says). The file is JSON, which the guide allows ("any binary file"); with a Constant profile it holds the level, with Automatic
  also the profile.
* Temporal Processing: Enable blends frames either side: "Frames to Blend" is how many each way, and "Frame Blending" "sets the variance
  allowed between regions in blended frames: higher values attempt to blend more regions". Each neighbour is moved onto the current
  frame along the `Motion` input's vectors (pixels per frame, scaled by the distance in frames, the guide's "pre-calculated motion
  vectors"), or along vectors NodeBased estimates between the two frames when none is wired ("Denoise can generate motion vectors
  internally"); where it then differs from the current frame by more than the blending (in any colour channel) its weight falls away
  (full up to the blending, none at twice it), and the frames are averaged with the current frame's weight 1 before the window average.

Foundry's own noise model has no equivalent here; the window average above stands in for it. What is left out is listed in
PARITY_2D_FIDELITY.md.

An older document's node keeps its own reading (`legacy_denoise` 1): the Gaussian bilateral of `bilateralknobs`, no luminance blend, no
output choice. A call that carries no flag reads as the older node.
"""
from __future__ import annotations

import json
import math

import numpy as np

TYPES = ("Digital", "Film")
MODELS = ("Constant", "Modulated")
PROFILES = ("Constant", "Automatic")
OUTPUTS = ("Result", "Noise")
NOMINAL_LEVEL = 1.0 / 3.0
MAX_RADIUS = 24
BANDS = 8
FREQUENCY_BANDS = ("high", "medium", "low", "very_low")
BAND_SIGMAS = (1.0, 2.0, 4.0)
FREQUENCY_REACH = 12                 # 3 sigma of the widest blur: what a tile must read past its edge for the bands
MIN_ANALYSIS = 80
FLAT_CURVE = '{"interpolation":"linear","points":[[0,1],[1,1]]}'
MAIN = {"denoise_type": "Digital", "denoise_model": "Modulated", "denoise_profile": "Constant", "denoise_output": "Result",
        "denoise_lift_blacks": 0, "denoise_rolloff": 2.0, "denoise_smoothness": 1.0, "denoise_lumablend": 0.7, "denoise_preserve_edges": 0}
TABS = {"denoise_luma_gain": 1.0, "denoise_chroma_gain": 5.0,
        "denoise_process_high": 1, "denoise_high_gain": 1.0, "denoise_process_medium": 1, "denoise_medium_gain": 1.0,
        "denoise_process_low": 1, "denoise_low_gain": 1.0, "denoise_process_very_low": 0, "denoise_very_low_gain": 1.0,
        "denoise_tune_profile": 0, "denoise_profile_low_gain": 1.0, "denoise_profile_mid_gain": 1.0, "denoise_profile_high_gain": 1.0,
        "denoise_profile_curve": FLAT_CURVE, "denoise_profile_frame": 1,
        "denoise_analysis_x": 0, "denoise_analysis_y": 0, "denoise_analysis_r": 0, "denoise_analysis_t": 0, "denoise_analysis_frame": 1,
        "denoise_lock_analysis": 0, "denoise_analysis_file": "", "denoise_analysis_level": 0.0, "denoise_analysis_profile": "",
        "denoise_frame_blending": 0.1, "denoise_frames": 1}
NEW = {"legacy_denoise": 0, **MAIN, **TABS}
# What a saved node gets on load: the flag and every new knob at the value that leaves it as it was (the luminance blend 1 keeps the
# denoised luminance, the model Constant the same threshold everywhere).
UPGRADE = {**NEW, "legacy_denoise": 1, "denoise_model": "Constant", "denoise_lumablend": 1.0}


def is_legacy(p):
    return bool(int(p.get("legacy_denoise", 1)))


def frequency_neutral(p):
    """True when every band is on at gain 1 (the bands then sum to the whole noise and need not be cut)."""
    return (all(int(p.get(f"denoise_process_{b}", 1 if b != "very_low" else 0)) == 1 for b in FREQUENCY_BANDS)
            and all(float(p.get(f"denoise_{b}_gain", 1.0)) == 1.0 for b in FREQUENCY_BANDS))


def uses_bands(p):
    return not is_legacy(p) and not frequency_neutral(p)


def radius(p):
    """The window's reach in pixels each way (what a tile must read past its edge)."""
    strength = max(0.0, float(p.get("denoise_strength", 0.2)))
    base = 2.0 + 5.0 * strength
    if is_legacy(p):
        return int(math.ceil(base))
    reach = int(math.ceil(base * max(0.0, float(p.get("denoise_smoothness", 1.0)))))
    return (min(MAX_RADIUS, reach) + (1 if int(p.get("denoise_preserve_edges", 0)) else 0)
            + (FREQUENCY_REACH if uses_bands(p) else 0))


def region_box(p):
    """(x, y, r, t) of the analysis box in pixels, or None when all four are 0 (no box)."""
    box = tuple(int(p.get(f"denoise_analysis_{n}", 0)) for n in "xyrt")
    return None if box == (0, 0, 0, 0) else box


def region_too_small(p):
    box = region_box(p)
    return box is not None and (box[2] - box[0] < MIN_ANALYSIS or box[3] - box[1] < MIN_ANALYSIS)


def is_locked(p):
    return not is_legacy(p) and bool(int(p.get("denoise_lock_analysis", 0))) and stored_analysis(p) is not None


def wants_analysis_frame(p):
    """True when the noise is measured in a box on a frame of the Source (so that frame may have to be read)."""
    return not is_legacy(p) and region_box(p) is not None and not region_too_small(p) and not is_locked(p)


def wants_profile_frame(p):
    return not is_legacy(p) and str(p.get("denoise_profile", "Constant")) == "Automatic" and not is_locked(p)


def temporal_frames(p):
    """The frames either side that temporal processing blends: how many each way, 0 when it is off."""
    if is_legacy(p) or not int(p.get("temporal", 0)):
        return 0
    return max(0, int(p.get("denoise_frames", 1)))


def needs_full_frame(p, has_noise_input=False):
    """The noise level is measured on a whole picture, so a node that measures it, reads other frames or blends them cannot be cut into tiles."""
    if is_legacy(p):
        return False
    return (has_noise_input or str(p.get("denoise_profile", "Constant")) == "Automatic" or region_box(p) is not None
            or bool(temporal_frames(p)))


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


def stored_analysis(p):
    """(level, profile levels or None) held by the node (Analyze Noise and Import set them), None when it holds none."""
    level = float(p.get("denoise_analysis_level", 0.0) or 0.0)
    text = str(p.get("denoise_analysis_profile", "") or "")
    profile = None
    if text:
        try:
            values = json.loads(text)
            if isinstance(values, list) and len(values) == BANDS:
                profile = np.asarray([float(v) for v in values], np.float64)
        except (ValueError, TypeError):
            profile = None
    if level <= 0.0 and profile is None:
        return None
    return level, profile


def _box_rows(image, p):
    """The pixels inside the analysis box (the whole picture with no box); the box's y counts from the bottom edge."""
    box = region_box(p)
    if box is None:
        return image
    x, y, r, t = box
    h, w = image.shape[:2]
    top, bottom = max(0, h - t), min(h, h - y)
    return image[top:bottom, max(0, x):min(w, r)]


def analysis_levels(analysis_image, profile_image, p, noise=None):
    """(constant level, profile levels) measured from the pictures the node's settings name: the Noise input when wired or the Source in the
    analysis box, and the Source's per-luminance profile on the profile frame."""
    measured = noise if noise is not None else analysis_image
    level = noise_level(_box_rows(measured, p)) if (region_box(p) is not None or noise is not None) else NOMINAL_LEVEL
    return level, noise_profile(profile_image)[1]


def analyze(analysis_image, profile_image, p, noise=None):
    """Analyze Noise: {"level", "profile"} for the node to hold."""
    level, profile = analysis_levels(analysis_image, profile_image, p, noise)
    return {"level": float(level), "profile": [float(v) for v in profile]}


def export_analysis(path, analysis, automatic):
    """Export: the Constant profile's analysis is its level; Automatic also saves the noise profile."""
    out = {"format": "nodebased-denoise-analysis", "level": float(analysis["level"])}
    if automatic:
        out["profile"] = [float(v) for v in analysis["profile"]]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(out, handle, indent=1)


def import_analysis(path):
    """Import: the {"level", "profile" (or None)} an exported file holds."""
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or data.get("format") != "nodebased-denoise-analysis":
        raise ValueError("not a NodeBased Denoise analysis file")
    profile = data.get("profile")
    if profile is not None and len(profile) != BANDS:
        raise ValueError(f"the analysis profile must hold {BANDS} levels")
    return {"level": float(data["level"]), "profile": None if profile is None else [float(v) for v in profile]}


def _profile_factor(lum, p):
    """The multiplier the Profile tab puts on the threshold at each luminance: the curve, and with Tune Profile on the three gains."""
    from . import colorcurves
    factor = np.ones(lum.shape, np.float64)
    text = str(p.get("denoise_profile_curve", FLAT_CURVE) or FLAT_CURVE)
    if text != FLAT_CURVE:
        curve = colorcurves.decode(text)
        xs = [point[0] for point in curve["points"]]
        factor = factor * np.asarray(colorcurves.evaluate_array(curve, np.clip(lum, xs[0], xs[-1])), np.float64)
    if int(p.get("denoise_tune_profile", 0)):
        gains = [float(p.get(f"denoise_profile_{n}_gain", 1.0)) for n in ("low", "mid", "high")]
        factor = factor * np.interp(lum, [0.0, 0.5, 1.0], gains)
    return factor


def threshold_map(image, p, noise=None, refs=None):
    """The colour threshold T at each pixel: 3 x noise level x amount, scaled by the model's brightness curve and the Profile tab."""
    refs = refs or {}
    amount = max(0.0, float(p.get("denoise_strength", 0.2)))
    lum = np.clip(luminance(image), 0.0, 1.0)
    profile = str(p.get("denoise_profile", "Constant"))
    held = stored_analysis(p) if (not is_legacy(p) and int(p.get("denoise_lock_analysis", 0))) else None
    if profile == "Automatic":
        if held is not None and held[1] is not None:
            centres, levels = 0.5 * (np.linspace(0, 1, BANDS + 1)[:-1] + np.linspace(0, 1, BANDS + 1)[1:]), held[1]
        else:
            centres, levels = noise_profile(refs.get("profile", image))
        level = np.interp(lum, centres, levels)
    else:
        if held is not None and held[0] > 0.0:
            level = held[0]
        elif region_box(p) is not None:
            measured = noise if noise is not None else refs.get("analysis", image)
            level = max(1e-4, noise_level(_box_rows(measured, p)))
        else:
            level = NOMINAL_LEVEL if noise is None else max(1e-4, noise_level(noise))
        level = np.full(lum.shape, level)
        if str(p.get("denoise_model", "Modulated")) == "Modulated":
            bright = 1.0 - lum if str(p.get("denoise_type", "Digital")) == "Film" else lum
            level = level * (0.25 + 1.5 * bright)
    return np.maximum(0.01, 3.0 * amount * level) * _profile_factor(lum, p)       # the 0.01 floor is the older node's; a profile gain of 0 still removes none


def _window_average(image, p, thresh):
    rad = radius(p) - (1 if int(p.get("denoise_preserve_edges", 0)) else 0) - (FREQUENCY_REACH if uses_bands(p) else 0)
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


def _channel_average(image, p, thresh, scale=1.0):
    """The window average with the luminance threshold scaled by Luminance Gain and the colour's by Chrominance Gain (and both by `scale`)."""
    luma_gain = max(0.0, float(p.get("denoise_luma_gain", 1.0))) * scale
    chroma_gain = max(0.0, float(p.get("denoise_chroma_gain", 5.0))) * scale
    floor = 1e-6
    by_luma = _window_average(image, p, np.maximum(floor, thresh * luma_gain))
    if luma_gain == chroma_gain:
        return by_luma
    by_chroma = _window_average(image, p, np.maximum(floor, thresh * chroma_gain))
    out = by_luma.copy()
    lum_l, lum_c = luminance(by_luma), luminance(by_chroma)
    colour = by_chroma[..., :3] - lum_c[..., None]
    out[..., :3] = lum_l[..., None] + colour
    return out


def _gauss(plane, sigma):
    radius_ = int(math.ceil(3.0 * sigma))
    taps = np.exp(-(np.arange(-radius_, radius_ + 1) ** 2) / (2.0 * sigma * sigma))
    taps /= taps.sum()
    h, w = plane.shape[:2]
    padded = np.pad(plane, ((radius_, radius_), (radius_, radius_)) + ((0, 0),) * (plane.ndim - 2), mode="edge")
    rows = sum(wt * padded[:, i:i + w] for i, wt in enumerate(taps))
    return sum(wt * rows[i:i + h] for i, wt in enumerate(taps))


def split_bands(noise):
    """The removed noise as (high, medium, low, very low); they sum to `noise`."""
    blurs = [_gauss(noise, s) for s in BAND_SIGMAS]
    return noise - blurs[0], blurs[0] - blurs[1], blurs[1] - blurs[2], blurs[2]


def tune_frequencies(image, p, thresh):
    """The denoised picture under the Tune Frequencies settings: each processed band of the removed noise comes from the window average at
    that band's gain."""
    passes = {}
    total = np.zeros(image.shape, np.float64)
    for index, name in enumerate(FREQUENCY_BANDS):
        if not int(p.get(f"denoise_process_{name}", 1 if name != "very_low" else 0)):
            continue
        gain = float(p.get(f"denoise_{name}_gain", 1.0))
        if gain not in passes:
            passes[gain] = split_bands(image.astype(np.float64) - _channel_average(image, p, thresh, gain))
        total += passes[gain][index]
    return image.astype(np.float64) - total


def analyze_node(evaluator, document, key, frame=1):
    """What Analyze Noise does for the Denoise node `key` of `document`: the params that hold its analysis (and lock it), measured on the
    Noise input when wired, else on the Source at the Analysis Frame, with the Automatic profile on the Profile Frame."""
    node = document["nodes"][key]
    p = {**NEW, **node["params"]}
    source_key = node["inputs"]["image"]
    noise_key = node["inputs"].get("noise")

    def picture(target, at):
        return evaluator.evaluate_raster(dict(document, view=target), target=target, frame=at).pixels
    analysis = picture(source_key, int(p["denoise_analysis_frame"]))
    profile = analysis if int(p["denoise_profile_frame"]) == int(p["denoise_analysis_frame"]) else picture(source_key, int(p["denoise_profile_frame"]))
    noise = picture(noise_key, frame) if noise_key else None
    found = analyze(analysis, profile, p, noise)
    return {"denoise_analysis_level": found["level"], "denoise_analysis_profile": json.dumps(found["profile"]), "denoise_lock_analysis": 1}


def held_analysis(p):
    """The {"level", "profile"} the node holds, for Export; None when it holds none."""
    held = stored_analysis(p)
    if held is None:
        return None
    level, profile = held
    return {"level": level, "profile": [float(v) for v in profile] if profile is not None else [level] * BANDS}


def imported_params(analysis):
    """The params Import sets: the analysis it read, and the lock ("activated by default if you use the Import button")."""
    out = {"denoise_analysis_level": float(analysis["level"]), "denoise_lock_analysis": 1}
    if analysis.get("profile") is not None:
        out["denoise_analysis_profile"] = json.dumps([float(v) for v in analysis["profile"]])
    return out


def blend_frames(image, neighbours, p, motion=None):
    """Temporal Processing: `image` averaged with each neighbour frame, moved onto it and weighted by how well it matches.

    `neighbours` is [(frames from this one (negative before), picture)]; `motion` the Motion input's vectors (pixels per frame, from this frame
    to the next) or None to estimate them."""
    if not neighbours:
        return image
    from .opticalflow import _sample, flow_pair
    h, w = image.shape[:2]
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    allowed = max(1e-6, float(p.get("denoise_frame_blending", 0.1)))
    total = image.astype(np.float64).copy()
    weights = np.ones((h, w, 1), np.float64)
    for offset, picture in neighbours:
        if motion is not None:
            vx, vy = motion[..., 0] * offset, motion[..., 1] * offset
        else:
            flow, _, _ = flow_pair(image, picture, vector_detail=4, smoothness=1.0, backend="cpu")
            vx, vy = np.asarray(flow)[..., 0], np.asarray(flow)[..., 1]
        moved = _sample(picture.astype(np.float32), xx + vx, yy + vy).astype(np.float64)
        miss = np.abs(moved[..., :3] - image[..., :3]).max(axis=2)
        weight = np.clip((2.0 * allowed - miss) / allowed, 0.0, 1.0)[..., None]
        total += moved * weight
        weights += weight
    return (total / weights).astype(np.float32)


def denoise(image, p, noise=None, refs=None):
    """Denoise on an (h, w, 4) picture; `noise` (any shape) is the picture the noise level is measured on, None for the nominal level.

    `refs` carries what other frames and inputs add: "analysis" and "profile" (the Source at those frames), "neighbours"
    ([(offset, picture)]) and "motion" (the Motion input's vectors)."""
    if is_legacy(p):
        from . import bilateralknobs
        strength = float(p.get("denoise_strength", 0.2))
        return bilateralknobs.bilateral(image, {"spatial_size": 2.0 + 5.0 * strength, "colour_sigma": max(0.01, strength)})
    refs = refs or {}
    source = image
    if refs.get("neighbours"):
        image = blend_frames(image, refs["neighbours"], p, refs.get("motion"))
    if float(p.get("denoise_strength", 0.2)) <= 0.0 or region_too_small(p):
        result = image.astype(np.float64)        # "A value of 0 removes no noise."; a box under 80 pixels analyzes nothing
    else:
        thresh = threshold_map(image, p, noise, refs)
        result = tune_frequencies(image, p, thresh) if uses_bands(p) else _channel_average(image, p, thresh)
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
        return (source.astype(np.float32) - result).astype(np.float32)
    return result


LIMITS = {"denoise_rolloff": (1.0, 10.0), "denoise_smoothness": (0.0, 4.0), "denoise_lumablend": (0.0, 1.0),
          "denoise_lift_blacks": (0, 1), "denoise_preserve_edges": (0, 1), "legacy_denoise": (0, 1),
          "denoise_luma_gain": (0.0, 20.0), "denoise_chroma_gain": (0.0, 20.0),
          **{f"denoise_{b}_gain": (0.0, 20.0) for b in FREQUENCY_BANDS}, **{f"denoise_process_{b}": (0, 1) for b in FREQUENCY_BANDS},
          "denoise_tune_profile": (0, 1), "denoise_profile_low_gain": (0.0, 20.0), "denoise_profile_mid_gain": (0.0, 20.0),
          "denoise_profile_high_gain": (0.0, 20.0), "denoise_profile_frame": (-100000, 100000), "denoise_analysis_frame": (-100000, 100000),
          **{f"denoise_analysis_{n}": (-100000, 100000) for n in "xyrt"}, "denoise_lock_analysis": (0, 1), "denoise_analysis_level": (0.0, 100.0),
          "denoise_frame_blending": (0.0, 1.0), "denoise_frames": (1, 5)}
CHOICES = {"denoise_type": list(TYPES), "denoise_model": list(MODELS), "denoise_profile": list(PROFILES), "denoise_output": list(OUTPUTS)}
