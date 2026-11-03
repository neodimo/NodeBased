"""Behaviour cases for EdgeBlur, GodRays, VolumeRays and BumpBoss (lane 2, 2D behaviour step G).

Each case states one behaviour of one node and compares NodeBased's kernel with a reference written here from that statement, in plain
float64 loops that share no code with NodeBased's own filters. `source` is "guide" when the Nuke 17.0 Reference Guide's text states the
behaviour (pages under .../reference_guide/filter_nodes/) and "convention" when only the form is long-established; the note says which.
What cannot be written down without guessing is `unscored`, with the reason.

Edge handling is stated nowhere for these nodes, so the cases that read neighbours compare the interior only (`valid` drops the pixels
whose neighbourhood leaves the 16 x 12 fixture).
"""
from __future__ import annotations

import math

import numpy as np

GUIDE = "https://learn.foundry.com/nuke/17.0/content/reference_guide/filter_nodes/"


def _window_extreme(plane, r, use_max):
    """Per-pixel max or min of the (2r + 1) square around it, edge pixels repeated; plain loops."""
    h, w = plane.shape
    out = np.zeros((h, w))
    for y in range(h):
        for x in range(w):
            vals = [plane[min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)] for dy in range(-r, r + 1) for dx in range(-r, r + 1)]
            out[y, x] = max(vals) if use_max else min(vals)
    return out


def _window_mean(img, r):
    """Per-pixel mean of the (2r + 1) square around it in every channel, edge pixels repeated; plain loops."""
    h, w, n = img.shape
    out = np.zeros((h, w, n))
    for y in range(h):
        for x in range(w):
            acc = np.zeros(n)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    acc += img[min(max(y + dy, 0), h - 1), min(max(x + dx, 0), w - 1)]
            out[y, x] = acc / float((2 * r + 1) ** 2)
    return out


def _interior(shape, reach):
    valid = np.zeros(shape, bool)
    if reach:
        valid[reach:-reach, reach:-reach, :] = True
    else:
        valid[...] = True
    return valid


def cases():
    from nuke_fidelity_cases import Case

    out = []

    def case(id, nuke, nuke_class, nuke_knobs, nb_kind, nb_params, expected, source, statement, unscored="", nuke_inputs=("A",),
             nb_inputs=("A",), inexpressible=""):
        out.append(Case(id=id, nuke=nuke, nuke_class=nuke_class, nuke_knobs=nuke_knobs, nuke_inputs=nuke_inputs, nb_kind=nb_kind,
                        nb_params=nb_params, nb_inputs=nb_inputs, expected=expected, source=source, statement=statement,
                        inexpressible=inexpressible, unscored=unscored))

    # ------------------------------------------------------------------------------------------------------ EdgeBlur
    def edge_band(src, width):
        """1 where the alpha's (2 * ceil(width) + 1) square holds both 0 and 1 (a hard matte's edge band), else 0; plain loops."""
        r = int(math.ceil(width)) if width >= 0.5 else 0
        alpha = src[..., 3]
        return _window_extreme(alpha, r, True) - _window_extreme(alpha, r, False)

    edge_note = ("EdgeBlur guide: 'Blurs detected edges within the matte specified in the matte channel'; 'edge mult ... Lower values produce "
                 "sharper edges and vice versa', read as a band of edge_mult * size pixels each side of the matte's edge (the band is where the "
                 "alpha's box neighbourhood of that radius holds both the matte's values, a hard matte's morphological gradient). Pixels beyond "
                 "the band are not edges, so the picture there is unchanged. The fixture E is a hard 6 x 6 matte on a textured plate")

    def far_from_edge(width):
        def expected(fx):
            src = fx["E"].astype(np.float64)
            band = edge_band(src, width)
            return src, np.repeat((band == 0)[..., None], 4, axis=2)
        return expected

    def in_band(radius, width, tint=1.0, brightness=1.0):
        def expected(fx):
            src = fx["E"].astype(np.float64)
            band = edge_band(src, width)
            blurred = _window_mean(src, radius)
            blurred[..., :3] *= tint
            blurred *= brightness
            valid = (band == 1)[..., None] & _interior(src.shape, radius)
            return blurred, valid
        return expected

    box_note = ("the guide lists box among the filters ('Box is the fastest to render, whereas Gaussian blur is the smoothest') and calls size the "
                "size of the blur; box is taken as the plain average of the (2 * size + 1) square, as the Blur cases do; edge pixels are not compared")
    for size, mult in ((2, 1.0), (1, 1.0), (2, 0.5)):
        width = size * mult
        case(f"edgeblur_far_from_edge_s{size}_m{mult:g}", "EdgeBlur", "EdgeBlur",
             (("size", float(size)), ("filter", "box"), ("edge_mult", mult)), "EdgeBlur",
             {"edgeblur_size": float(size), "kernel_filter": "box", "edge_mult": mult, "quality": 15.0, "crop": 1, "channels": "rgba"},
             far_from_edge(width), "guide", f"EdgeBlur size {size}, edge mult {mult:g}: pixels outside the edge band are unchanged. " + edge_note,
             nuke_inputs=("E",), nb_inputs=("E",))
    for size in (1, 2):
        case(f"edgeblur_band_is_blur_s{size}", "EdgeBlur", "EdgeBlur", (("size", float(size)), ("filter", "box"), ("edge_mult", 1.0)), "EdgeBlur",
             {"edgeblur_size": float(size), "kernel_filter": "box", "edge_mult": 1.0, "quality": 15.0, "crop": 1, "channels": "rgba"},
             in_band(size, size * 1.0), "guide",
             f"EdgeBlur size {size}: inside the edge band (where the matte's neighbourhood holds both values) the picture is the blur of the same "
             f"size, as a plain Blur gives it. " + edge_note + "; " + box_note, nuke_inputs=("E",), nb_inputs=("E",))
    case("edgeblur_tint_multiplies_rgb", "EdgeBlur", "EdgeBlur", (("size", 1.0), ("filter", "box"), ("edge_mult", 1.0), ("tint", 0.5)), "EdgeBlur",
         {"edgeblur_size": 1.0, "kernel_filter": "box", "edge_mult": 1.0, "edgeblur_tint": 0.5, "quality": 15.0, "crop": 1, "channels": "rgba"},
         in_band(1, 1.0, tint=0.5), "guide",
         "EdgeBlur tint 0.5: 'the multiplier for the rgb channels in the edge blur', so inside the band the red, green and blue of the blur are "
         "halved and the alpha is not. " + box_note, nuke_inputs=("E",), nb_inputs=("E",))
    case("edgeblur_brightness_multiplies_all", "EdgeBlur", "EdgeBlur", (("size", 1.0), ("filter", "box"), ("edge_mult", 1.0), ("brightness", 2.0)),
         "EdgeBlur", {"edgeblur_size": 1.0, "kernel_filter": "box", "edge_mult": 1.0, "edgeblur_brightness": 2.0, "quality": 15.0, "crop": 1,
                      "channels": "rgba"},
         in_band(1, 1.0, brightness=2.0), "guide",
         "EdgeBlur brightness 2: 'the multiplier for all channels in the edge blur', so inside the band every channel of the blur, the alpha "
         "included, is doubled. " + box_note, nuke_inputs=("E",), nb_inputs=("E",))
    case("edgeblur_size_zero_is_identity", "EdgeBlur", "EdgeBlur", (("size", 0.0), ("filter", "box")), "EdgeBlur",
         {"edgeblur_size": 0.0, "kernel_filter": "box", "edge_mult": 2.0, "quality": 15.0, "crop": 1, "channels": "rgba"},
         lambda fx: (fx["E"].astype(np.float64), np.ones(fx["E"].shape, bool)), "convention",
         "EdgeBlur size 0: a blur of size 0 changes nothing, so the picture is unchanged (the long-established reading of a zero-size blur)",
         nuke_inputs=("E",), nb_inputs=("E",))
    case("edgeblur_gaussian_weights", "EdgeBlur", "EdgeBlur", (("size", 2.0), ("edge_mult", 1.0)), "EdgeBlur",
         {"edgeblur_size": 2.0, "kernel_filter": "gaussian", "edge_mult": 1.0, "quality": 15.0, "crop": 1, "channels": "rgba"},
         in_band(2, 2.0), "convention", "EdgeBlur gaussian (the default filter): the weights for a given size",
         unscored="the guide does not state the gaussian's sigma for a given size (only that gaussian is the default and the smoothest), so its "
                  "weights cannot be written down without guessing; tests/test_2d_behaviour_extra_filter.py checks that the band equals a plain "
                  "Blur of the same filter and size", nuke_inputs=("E",), nb_inputs=("E",))
    case("edgeblur_quality_scaled", "EdgeBlur", "EdgeBlur", (("size", 6.0), ("filter", "box"), ("quality", 2.0)), "EdgeBlur",
         {"edgeblur_size": 6.0, "kernel_filter": "box", "edge_mult": 1.0, "quality": 2.0, "crop": 1, "channels": "rgba"},
         in_band(6, 6.0), "guide", "EdgeBlur: a size above quality scales the image down, filters, and scales up with linear interpolation",
         unscored="the guide does not give the scale factor, the down-scaling filter or the block alignment, so the pixels cannot be written "
                  "down; tests check that the result is bounded and tile-exact", nuke_inputs=("E",), nb_inputs=("E",))

    # ------------------------------------------------------------------------------------------------------- GodRays
    rays_note = ("GodRays guide: 'Averages together many samples of the image between its initial position and the translate and center "
                 "indicators'; scale 'Controls the direction of zoom in relation to the center widget. Values less than one zoom out'. The "
                 "samples sit at equal fractions t = k / steps (k = 0 to steps) of the way from the initial picture (t = 0) to the full "
                 "transform (t = 1), the endpoints being the guide's 'initial' and 'final' image; the spacing is the long-established reading. "
                 "The fixture P is one white pixel at x 12 and y 6 and the centre sits at x 4 and y 6, so the pixel's distance 8 becomes 8 * scale")
    rays = {"legacy_godrays": 0, "steps": 2, "center_x": 4.0, "center_y": 6.0, "rays_scale": 0.5, "rays_max": 1, "rays_translate_x": 0.0,
            "rays_translate_y": 0.0, "rays_rotate": 0.0, "rays_skew": 0.0, "rays_from_color": 1.0, "rays_to_color": 1.0, "rays_gamma": 1.0}
    rays_knobs = (("center", "4 6"), ("scale", 0.5), ("steps", 2.0), ("max", True))

    def streak(values):
        """Zeros with `values[(x, channels)]` at row 6; valid at the pixels named (all channels, or the channels given)."""
        def expected(fx):
            ref = np.zeros(fx["P"].shape, np.float64)
            valid = np.zeros(fx["P"].shape, bool)
            for x, channels, value in values:
                for c in channels:
                    ref[6, x, c] = value
                    valid[6, x, c] = True
            return ref, valid
        return expected

    def everywhere(fx):
        return np.zeros(fx["P"].shape, np.float64), np.ones(fx["P"].shape, bool)

    def streak_all(fx):
        ref, _ = everywhere(fx)
        for x in (12, 10, 8):
            ref[6, x, :] = 1.0
        return ref, np.ones(ref.shape, bool)

    case("godrays_default_is_identity", "GodRays", "GodRays", (("steps", 5.0),), "GodRays",
         {"legacy_godrays": 0, "steps": 5}, lambda fx: (fx["A"].astype(np.float64), np.ones(fx["A"].shape, bool)), "guide",
         "GodRays at its defaults (translate 0 0, rotate 0, scale 1, skew 0, from and to colour 1): every sample is the unmoved picture, so the "
         "average of the samples is the picture")
    case("godrays_max_streaks_along_the_ray", "GodRays", "GodRays", rays_knobs, "GodRays", dict(rays), streak_all, "convention",
         "GodRays with max on, scale 0.5 and steps 2: the white pixel appears at its initial position and at the samples toward the centre "
         "(x 12, 10 and 8 on the row of the centre), at full strength, and nowhere else. " + rays_note, nuke_inputs=("P",), nb_inputs=("P",))
    case("godrays_from_and_to_colour_at_the_ends", "GodRays", "GodRays", rays_knobs + (("from_color", 0.5), ("to_color", 2.0)), "GodRays",
         {**rays, "rays_from_color": 0.5, "rays_to_color": 2.0},
         streak([(12, (0, 1, 2), 0.5), (8, (0, 1, 2), 2.0)]), "guide",
         "GodRays from color 0.5 and to color 2: from color 'sets the color by which the initial image is multiplied' and to color 'the color by "
         "which the final image is multiplied', so the white pixel is 0.5 where it started and 2 at the last sample (x 12 and x 8); the "
         "samples between are not scored. " + rays_note, nuke_inputs=("P",), nb_inputs=("P",))
    case("godrays_average_weights", "GodRays", "GodRays", (("center", "4 6"), ("scale", 0.5), ("steps", 2.0)), "GodRays",
         {**rays, "rays_max": 0}, everywhere, "convention", "GodRays with max off: each sample's share of the average",
         unscored="the guide says 'averages together many samples' without saying whether steps counts the samples or the gaps between them "
                  "(2 or 3 samples at steps 2), so the weight of each streak pixel cannot be written down without guessing; tests check that "
                  "the average is the mean of the same samples taken singly", nuke_inputs=("P",), nb_inputs=("P",))
    for knob, nb_name, value in (("translate", "rays_translate_x", 3.0), ("rotate", "rays_rotate", 20.0), ("skew", "rays_skew", 0.4)):
        case(f"godrays_{knob}_direction", "GodRays", "GodRays", (("center", "4 6"), (knob, value if knob != "translate" else "3 0"), ("steps", 2.0), ("max", True)),
             "GodRays", {**rays, "rays_scale": 1.0, nb_name: value}, everywhere, "convention",
             f"GodRays {knob}: the direction and sign of the movement",
             unscored=f"the guide names {knob} but not its sign (clockwise or not, up or down; Nuke's y runs up), the order it combines with the "
                      "other controls (its note says translate and skew do not merge correctly with scale), or its units; tests check that the "
                      "samples move along the ray the knob names and that the knob acts",
             nuke_inputs=("P",), nb_inputs=("P",))
    case("godrays_gamma_space", "GodRays", "GodRays", rays_knobs + (("from_color", 0.5), ("to_color", 2.0), ("gamma", 2.2)), "GodRays",
         {**rays, "rays_from_color": 0.5, "rays_to_color": 2.0, "rays_gamma": 2.2}, everywhere, "convention",
         "GodRays gamma: the colours are interpolated in the gamma space", unscored="the guide gives gamma one sentence ('the gamma space in "
         "which colors are interpolated') and no formula; the ends are unchanged by it and the middle sample is not scored",
         nuke_inputs=("P",), nb_inputs=("P",))

    return out
