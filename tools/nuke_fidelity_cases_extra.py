"""Behaviour cases for EdgeBlur, GodRays, VolumeRays and BumpBoss (lane 2, 2D behaviour step G) and Flare, Glint, Sparkles and ScannedGrain (step H).

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


    # ---------------------------------------------------------------------------------------------------- VolumeRays
    case("volumerays_rays_radiate_from_the_volume_centre", "VolumeRays", "VolumeRays", (("vol_pos", "4 6"),), "VolumeRays",
         {"center_x": 4.0, "center_y": 6.0, "comp_me": 0}, lambda fx: (np.zeros(fx["P"].shape, np.float64), np.ones(fx["P"].shape, bool)), "guide",
         "VolumeRays: 'ray lighting effects radiating from the vol_pos indicator'", unscored="the node is a gizmo (a group of nodes) whose rays are "
         "built from the image's luminance or alpha through a style, a noise flicker, a radial falloff and a ray length in the gizmo's own units, none "
         "of which the guide states as a formula; NodeBased's rays come from the image colour weighted by a wired matte, so a pixel reference for "
         "either cannot be written down without guessing (tests/test_2d_behaviour_extra_filter.py checks that the rays lie on the line through the "
         "volume centre)", nuke_inputs=("P",), nb_inputs=("P", "P"))

    # ------------------------------------------------------------------------------------------------------ BumpBoss
    case("bumpboss_emboss_of_b_by_a", "BumpBoss", "BumpBoss", (("lightposition", "480 360"), ("center", "320 240")), "BumpBoss",
         {"light_angle": 135.0}, lambda fx: (fx["B"].astype(np.float64), np.zeros(fx["B"].shape, bool)), "guide",
         "BumpBoss: 'an emboss effect by overlaying input A over input B'; height zero 'is level with the image, producing no emboss'",
         unscored="the guide states what each knob does and no lighting formula (how A's slope, the light's offset from the centre, height, "
                  "intensity and the minimum shadow combine into B's pixels), and NodeBased's BumpBoss is a one-input emboss lit from an angle; the "
                  "node's remaining gap is recorded as unsupported knobs in docs/parity/nodebased_to_nuke.json", nuke_inputs=("A", "B"), nb_inputs=("A",))


    # =========================================================================================== step H: the Draw group's four effects
    draw_url = "https://learn.foundry.com/nuke/17.0/content/reference_guide/draw_nodes/"
    H, W = 12, 16

    def luma709(rgb):
        return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]

    # ------------------------------------------------------------------------------------------------------------- Flare
    flare_note = ("Flare guide (" + draw_url + "flare.html): 'Nuke flares are implemented as a donut-like shape' with an inner, a middle and an outer radius; "
                  "'Flares fade from black at the outer radius to the ring color on the middle radius' and 'from the ring color on the middle radius to "
                  "the inner color on the inner radius'; the inner color fills the hole. The flare is added to the picture ('the image sequence to which "
                  "you want to add lens flares'). The fixture P is one white pixel at x 12 and y 6, the flare sits at x 4 and y 6 with radii 1, 3 and 5, "
                  "so the pixels at exactly those distances (0, 1 and 3 along the axes, 5 on the 3-4-5 diagonals) are the stated landmarks; the pixels "
                  "between them carry the fade, whose curve the guide does not give, and are not compared; the alpha is not compared")
    flare_nb = {"legacy_flare": 0, "position_x": 4.0, "position_y": 6.0, "flare_radius_inner": 1.0, "flare_radius_middle": 3.0, "flare_radius_outer": 5.0,
                "flare_size_mult": 1.0, "flare_ring_color": 1.0, "flare_inner_color": 0.5, "brightness": 1.0}
    flare_knobs = (("position", "4 6"), ("radius", "1 3 5"), ("size_mult", 1.0), ("ring_color", 1.0), ("inner_color", 0.5), ("brightness", 1.0))

    def flare_expected(cx, cy, inner, middle, outer, ring, core, brightness, landmarks):
        """P plus the donut at the stated landmarks: `landmarks` picks which of 'ring', 'outside' and 'core' are compared."""
        def expected(fx):
            src = fx["P"].astype(np.float64)
            ref, valid = src.copy(), np.zeros(src.shape, bool)
            for y in range(H):
                for x in range(W):
                    d = math.hypot(x - cx, y - cy)
                    if "core" in landmarks and d <= inner + 1e-9:
                        value = core
                    elif "ring" in landmarks and abs(d - middle) < 1e-9:
                        value = ring
                    elif "outside" in landmarks and d >= outer - 1e-9:
                        value = 0.0
                    else:
                        continue
                    ref[y, x, :3] += brightness * value
                    valid[y, x, :3] = True
            return ref, valid
        return expected

    def flare_case(id, nb_over, knob_over, expected, statement, **kw):
        nb = {**flare_nb, **nb_over}
        knobs = {**dict(flare_knobs), **knob_over}
        case(id, "Flare", "Flare", tuple(knobs.items()), "Flare", nb, expected, kw.pop("source", "guide"), statement + ". " + flare_note,
             nuke_inputs=("P",), nb_inputs=("P",), **kw)

    flare_case("flare_ring_color_at_the_middle_radius", {"flare_ring_color": 0.8}, {"ring_color": 0.8},
               flare_expected(4, 6, 1, 3, 5, 0.8, 0.5, 1.0, ("ring",)), "Flare ring color 0.8: the pixels at the middle radius hold the ring colour")
    flare_case("flare_black_from_the_outer_radius", {"flare_ring_color": 0.8}, {"ring_color": 0.8},
               flare_expected(4, 6, 1, 3, 5, 0.8, 0.5, 1.0, ("outside",)), "Flare: from the outer radius outwards the flare is black, the picture unchanged")
    flare_case("flare_inner_color_fills_the_hole", {"flare_inner_color": 0.3}, {"inner_color": 0.3},
               flare_expected(4, 6, 1, 3, 5, 1.0, 0.3, 1.0, ("core",)), "Flare inner color 0.3: within the inner radius the flare holds the inner colour")
    flare_case("flare_brightness_scales_the_flare", {"flare_ring_color": 0.5, "brightness": 2.0}, {"ring_color": 0.5, "brightness": 2.0},
               flare_expected(4, 6, 1, 3, 5, 0.5, 0.5, 2.0, ("ring", "core", "outside")), "Flare brightness 2: 'adjusts the overall brightness of the flare', so the landmarks double")
    flare_case("flare_brightness_zero_is_invisible", {"brightness": 0.0}, {"brightness": 0.0},
               lambda fx: (fx["P"].astype(np.float64), np.repeat(np.concatenate([np.ones((H, W, 3), bool), np.zeros((H, W, 1), bool)], axis=2), 1, axis=0)),
               "Flare brightness 0: 'setting this value to 0 makes the flare invisible', so the picture is unchanged")
    flare_case("flare_size_mult_scales_the_radii_together", {"flare_radius_inner": 0.5, "flare_radius_middle": 1.5, "flare_radius_outer": 2.5, "flare_size_mult": 2.0,
                                                           "flare_ring_color": 0.8},
               {"radius": "0.5 1.5 2.5", "size_mult": 2.0, "ring_color": 0.8},
               flare_expected(4, 6, 1, 3, 5, 0.8, 0.5, 1.0, ("ring", "core", "outside")),
               "Flare size mult 2 on radii 0.5, 1.5 and 2.5: 'scales the flare globally, while retaining the ratio described by the radius controls', so it is the flare of radii 1, 3 and 5")
    flare_case("flare_position_is_the_centre", {"position_x": 9.0, "position_y": 3.0, "flare_ring_color": 0.8}, {"position": "9 3", "ring_color": 0.8},
               flare_expected(9, 3, 1, 3, 5, 0.8, 0.5, 1.0, ("ring", "core", "outside")),
               "Flare position 9 3: 'sets the center position of the flare', so the same landmarks sit around x 9 and y 3")
    for id, why in (("flare_fade_curve", "the fade between the radii (outer falloff and inner falloff 1 'sets the blending or falloff'): the guide gives no curve, so the pixels between the radii cannot be written down without guessing; tests check that the fade is monotonic and tile-exact"),
                    ("flare_anamorph", "anamorph 'simulates anamorphic lens flares by altering the default 1:1 flare shape' with no formula for the stretch"),
                    ("flare_chroma_shift", "chroma shift 'moves color components in the direction specified in the angle control' with no distance or sign convention given; NodeBased moves red along the angle and blue the opposite way"),
                    ("flare_alpha", "the guide does not say what a flare does to the alpha channel")):
        flare_case(id, {}, {}, lambda fx: (fx["P"].astype(np.float64), np.zeros(fx["P"].shape, bool)), "Flare: " + id.split("_", 1)[1].replace("_", " "),
                   source="convention", unscored=why)

    # ------------------------------------------------------------------------------------------------------------- Glint
    glint_note = ("Glint guide (" + draw_url + "glint.html): 'star-shaped rays around all the bright points'; no. of rays is 'the number of rays coming out of the "
                  "bright points ... if you want five-pointed stars, enter 5'; length 'sets the length of the rays'; from color is 'the color in the beginning "
                  "of the rays near the center point of the stars' and to color 'the color in the end of the rays'; steps 'how many steps the rays are formed "
                  "of'; max 'the brightest image forming the rays is used', off 'the images forming the rays are added up'; rotation 'rotates the "
                  "star-shapes by degrees. Negative values produce anti-clockwise rotation'. Read as: the rays are 360 / rays degrees apart, the first "
                  "pointing up at rotation 0 (the long-established upright star); copy k of steps (k = 1 to steps) of the highlights sits k / steps of the "
                  "length along a ray, its colour running from from color at the first copy to to color at the last; the rays are added to the picture. "
                  "The fixture P is one white pixel (value 1, above a tolerance of 0) at x 12 and y 6; the alpha is not compared")
    glint_nb = {"legacy_glint": 0, "rays": 4, "length": 4.0, "rotation": 0.0, "tolerance": 0.0, "glint_steps": 4, "glint_max": 1, "glint_from_color": 1.0,
                "glint_to_color": 0.0, "glint_oddlen": 1.0, "glint_effect_only": 0}
    glint_knobs = {"repeat": 4.0, "length": 4.0, "rotation": 0.0, "tolerance": 0.0, "steps": 4.0, "max": True, "from_color": 1.0, "to_color": 0.0}

    def glint_expected(rays, length, rotation, steps, start, end, tolerance=0.0, brightest=True, effect_only=False, oddlen=1.0):
        """The stated star on P, with copies placed by bilinear weights (a plain splat, shared with nothing in NodeBased)."""
        def expected(fx):
            src = fx["P"].astype(np.float64)
            effect = np.zeros((H, W, 3))
            for y0 in range(H):
                for x0 in range(W):
                    value = np.maximum(src[y0, x0, :3] - tolerance, 0.0)
                    if not value.any():
                        continue
                    for j in range(rays):
                        theta = math.radians(rotation + 360.0 * j / rays)
                        reach = length * (oddlen if j % 2 == 1 else 1.0)
                        for k in range(1, steps + 1):
                            colour = start + (end - start) * ((k - 1) / (steps - 1) if steps > 1 else 0.0)
                            px, py = x0 + math.sin(theta) * k / steps * reach, y0 - math.cos(theta) * k / steps * reach
                            fx0, fy0 = math.floor(px), math.floor(py)
                            for oy in (0, 1):
                                for ox in (0, 1):
                                    xi, yi = fx0 + ox, fy0 + oy
                                    weight = (1 - abs(px - xi)) * (1 - abs(py - yi))
                                    if 0 <= xi < W and 0 <= yi < H and weight > 0:
                                        add = value * colour * weight
                                        effect[yi, xi] = np.maximum(effect[yi, xi], add) if brightest else effect[yi, xi] + add
            ref = src.copy()
            ref[..., :3] = effect if effect_only else src[..., :3] + effect
            valid = np.ones(src.shape, bool)
            valid[..., 3] = False
            return ref, valid
        return expected

    def glint_case(id, nb_over, knob_over, expected, statement, **kw):
        nb = {**glint_nb, **nb_over}
        knobs = {**glint_knobs, **knob_over}
        case(id, "Glint", "Glint", tuple(knobs.items()), "Glint", nb, expected, kw.pop("source", "guide"), statement + ". " + glint_note,
             nuke_inputs=("P",), nb_inputs=("P",), **kw)

    glint_case("glint_four_rays", {}, {}, glint_expected(4, 4.0, 0.0, 4, 1.0, 0.0), "Glint no. of rays 4, length 4, 4 steps: four rays, up, right, down and left, "
               "each of four copies fading from 1 to 0")
    glint_case("glint_default_two_rays", {"rays": 2}, {"repeat": 2.0}, glint_expected(2, 4.0, 0.0, 4, 1.0, 0.0), "Glint no. of rays 2 (the default): two rays, up and down")
    glint_case("glint_single_ray_up", {"rays": 1}, {"repeat": 1.0}, glint_expected(1, 4.0, 0.0, 4, 1.0, 0.0), "Glint no. of rays 1: one ray, pointing up", source="convention")
    for turn, direction in ((90.0, "right (clockwise from up)"), (-90.0, "left (anti-clockwise from up)"), (180.0, "down")):
        glint_case(f"glint_rotation_{int(turn)}", {"rays": 1, "rotation": turn}, {"repeat": 1.0, "rotation": turn},
                   glint_expected(1, 4.0, turn, 4, 1.0, 0.0), f"Glint rotation {turn:g} on one ray: the ray points {direction}")
    glint_case("glint_length", {"length": 6.0, "glint_steps": 3}, {"length": 6.0, "steps": 3.0}, glint_expected(4, 6.0, 0.0, 3, 1.0, 0.0),
               "Glint length 6, 3 steps: the copies sit 2, 4 and 6 pixels out along each ray")
    glint_case("glint_from_and_to_colour", {"glint_from_color": 2.0, "glint_to_color": 0.5, "glint_steps": 3, "length": 3.0},
               {"from_color": 2.0, "to_color": 0.5, "steps": 3.0, "length": 3.0}, glint_expected(4, 3.0, 0.0, 3, 2.0, 0.5),
               "Glint from color 2, to color 0.5, 3 steps: the copies are 2, 1.25 and 0.5 times the highlight, first to last")
    glint_case("glint_max_off_adds_the_steps", {"rays": 1, "length": 2.0, "glint_max": 0, "glint_steps": 4}, {"repeat": 1.0, "length": 2.0, "max": False, "steps": 4.0},
               glint_expected(1, 2.0, 0.0, 4, 1.0, 0.0, brightest=False), "Glint max off: the copies add up where they overlap (copies half a pixel apart share pixels)")
    glint_case("glint_max_on_takes_the_brightest", {"rays": 1, "length": 2.0, "glint_max": 1, "glint_steps": 4}, {"repeat": 1.0, "length": 2.0, "max": True, "steps": 4.0},
               glint_expected(1, 2.0, 0.0, 4, 1.0, 0.0, brightest=True), "Glint max on: where copies overlap the brightest is used")
    glint_case("glint_tolerance_excludes_dim_pixels", {"tolerance": 2.0}, {"tolerance": 2.0},
               lambda fx: (fx["P"].astype(np.float64), np.concatenate([np.ones((H, W, 3), bool), np.zeros((H, W, 1), bool)], axis=2)),
               "Glint tolerance 2: 'only pixels above the threshold will bloom', the white pixel is 1, so the picture is unchanged")
    glint_case("glint_effect_only", {"glint_effect_only": 1}, {"effect_only": True}, glint_expected(4, 4.0, 0.0, 4, 1.0, 0.0, effect_only=True),
               "Glint effect only: 'only outputs the effect without merging it with the original input image', so the highlight itself is gone and only the rays remain")
    glint_case("glint_odd_ray_length", {"glint_oddlen": 0.5}, {"oddlen": 0.5}, glint_expected(4, 4.0, 0.0, 4, 1.0, 0.0, oddlen=0.5),
               "Glint odd ray length 0.5: every other ray is half as long", source="convention",
               unscored="the guide says odd ray length 'gives every other ray a different length and sets that length' without saying whether the first or the second ray "
                        "is the odd one, or whether the value is a multiple or a pixel length; tests check that it shortens the odd rays")
    glint_case("glint_highlight_strength", {"tolerance": 0.5}, {"tolerance": 0.5}, glint_expected(4, 4.0, 0.0, 4, 1.0, 0.0, tolerance=0.5), "Glint tolerance 0.5: the strength of a ray from a highlight above it",
               source="convention", unscored="the guide says only that pixels above the threshold bloom; whether a ray carries the pixel's value or only the part above the threshold is not stated, "
               "so the ray values cannot be written down without guessing (NodeBased carries the part above)")

    # --------------------------------------------------------------------------------------------------------- Sparkles
    spark_note = ("Sparkles guide (" + draw_url + "sparkles.html): 'creates customizable rays or sparkles from a central position indicator'; size 'sets the size "
                  "of the rays, in pixels, and adjusts the ray bounding box accordingly'; rays Nb 'sets the number of rays emitted'; thickness 'sets the thickness of "
                  "the emitted rays in pixels'; rotation 'sets the amount of rotation applied to the rays. Negative values produce clockwise rotation'; brightness "
                  "'sets ray brightness'; seed 'sets the random seed that controls ray characteristics'. Read as: each ray is a line from the position out to size "
                  "pixels, a pixel being lit only within thickness / 2 + 1 pixels of the line, the ray fading away from the origin with the ray falloff, "
                  "overlapping rays keeping the larger value, added to the picture. The seed chooses the directions: ray i points at rotation + 360 * (i + u_i) / "
                  "rays degrees anticlockwise with u from numpy's RandomState(seed).uniform(0, 1, rays), the one rule here that is ours; so the cases compare only "
                  "the stated landmarks: the origin, beyond the size, off every ray, and the unchanged picture. The fixture P is one white pixel far from the rays; "
                  "the alpha is not compared")
    spark_nb = {"legacy_sparkles": 0, "spark_x": 4.0, "spark_y": 6.0, "spark_size": 6.0, "spark_rotation": 0.0, "spark_ray_nb": 3, "spark_thick": 0.2,
                "spark_ray_falloff": 2.0, "spark_brightness": 1.0, "spark_color": 1.0, "seed": 23}
    spark_knobs = {"position": "4 6", "size": 6.0, "rotation": 0.0, "ray_nb": 3.0, "ray_thick": 0.2, "ray_falloff": 2.0, "seed": 23.0, "brightness": 1.0, "color": 1.0}
    rgb_valid = lambda: np.concatenate([np.ones((H, W, 3), bool), np.zeros((H, W, 1), bool)], axis=2)

    def spark_geometry(seed, rays, rotation, size, thick):
        """Per pixel: (lit by some ray, the nearest ray's along-distance); plain loops over the stated rule."""
        us = np.random.RandomState(seed).uniform(0.0, 1.0, rays) if rays else []
        angles = [math.radians(rotation + 360.0 * (i + us[i]) / rays) for i in range(rays)]
        lit = np.zeros((H, W), bool)
        for y in range(H):
            for x in range(W):
                for a in angles:
                    dx, dy = x - 4.0, y - 6.0
                    along = dx * math.cos(a) - dy * math.sin(a)
                    across = abs(-dx * math.sin(a) - dy * math.cos(a))
                    if 0.0 <= along <= size and across < thick / 2.0 + 1.0:
                        lit[y, x] = True
        return lit

    def spark_unlit(seed=23, rays=3, rotation=0.0, size=6.0, thick=0.2):
        def expected(fx):
            src = fx["P"].astype(np.float64)
            lit = spark_geometry(seed, rays, rotation, size, thick)
            valid = rgb_valid() & ~lit[..., None]
            return src.copy(), valid
        return expected

    def spark_case(id, nb_over, knob_over, expected, statement, **kw):
        nb = {**spark_nb, **nb_over}
        knobs = {**spark_knobs, **knob_over}
        case(id, "Sparkles", "Sparkles", tuple(knobs.items()), "Sparkles", nb, expected, kw.pop("source", "guide"), statement + ". " + spark_note,
             nuke_inputs=("P",), nb_inputs=("P",), **kw)

    def spark_origin(brightness, color):
        def expected(fx):
            src = fx["P"].astype(np.float64)
            ref, valid = src.copy(), np.zeros(src.shape, bool)
            ref[6, 4, :3] += brightness * color
            valid[6, 4, :3] = True
            return ref, valid
        return expected

    def spark_beyond(size):
        def expected(fx):
            src = fx["P"].astype(np.float64)
            valid = np.zeros(src.shape, bool)
            for y in range(H):
                for x in range(W):
                    if math.hypot(x - 4.0, y - 6.0) > size + 1e-9:
                        valid[y, x, :3] = True
            return src.copy(), valid
        return expected

    spark_case("sparkles_origin_holds_the_ray_colour", {"spark_ray_nb": 1, "spark_brightness": 1.5, "spark_color": 0.8},
               {"ray_nb": 1.0, "brightness": 1.5, "color": 0.8}, spark_origin(1.5, 0.8),
               "Sparkles with one ray, brightness 1.5 and color 0.8: the position itself, where every ray starts, holds brightness times color")
    for size in (4.0, 6.0):
        spark_case(f"sparkles_nothing_beyond_size_{int(size)}", {"spark_size": size, "spark_ray_nb": 50}, {"size": size, "ray_nb": 50.0}, spark_beyond(size),
                   f"Sparkles size {size:g}, 50 rays: beyond {size:g} pixels from the position the picture is unchanged (the ray bounding box)")
    spark_case("sparkles_no_rays_is_identity", {"spark_ray_nb": 0}, {"ray_nb": 0.0},
               lambda fx: (fx["P"].astype(np.float64), rgb_valid()), "Sparkles with 0 rays: nothing is emitted, the picture is unchanged")
    spark_case("sparkles_zero_brightness_is_identity", {"spark_brightness": 0.0}, {"brightness": 0.0},
               lambda fx: (fx["P"].astype(np.float64), rgb_valid()), "Sparkles brightness 0: the rays have no brightness, the picture is unchanged", source="convention")
    spark_case("sparkles_off_the_rays_is_unchanged", {}, {}, spark_unlit(), "Sparkles, 3 rays at seed 23: pixels farther than the thickness from every ray are unchanged",
               source="convention")
    spark_case("sparkles_other_seed_other_rays", {"seed": 5}, {"seed": 5.0}, spark_unlit(seed=5), "Sparkles at seed 5: the rays point elsewhere, and the pixels off them are unchanged",
               source="convention")
    spark_case("sparkles_rotation_turns_the_rays", {"spark_rotation": 37.0}, {"rotation": 37.0}, spark_unlit(rotation=37.0),
               "Sparkles rotation 37: the rays turn 37 degrees anticlockwise about the position, and the pixels off them are unchanged", source="convention")
    spark_case("sparkles_thickness_widens_the_rays", {"spark_thick": 3.0}, {"ray_thick": 3.0}, spark_unlit(thick=3.0),
               "Sparkles thickness 3: the rays are wider, and the pixels farther than the new width from every ray are unchanged", source="convention")
    for id, why in (("sparkles_ray_fade", "ray falloff 'sets how quickly rays falloff as distance from the position indicator increases': the guide gives no curve, so the values along a ray "
                                           "cannot be written down without guessing; NodeBased fades them as (1 - d / size) to the power of the falloff, and tests check that they never rise with distance"),
                    ("sparkles_modes", "the Sparkle type (Rays, Sparkles, Fireworks) changes how rays fade and break; only Rays is implemented, and the guide gives no formulae for the other two"),
                    ("sparkles_random_knobs", "the random thickness, broken rays, crawl, sparks and chroma knobs 'randomize' within ranges the guide does not state; none is implemented")):
        spark_case(id, {}, {}, lambda fx: (fx["P"].astype(np.float64), np.zeros(fx["P"].shape, bool)), "Sparkles: " + id.split("_", 1)[1].replace("_", " "),
                   source="convention", unscored=why)

    # --------------------------------------------------------------------------------------------------------- ScannedGrain
    grain_note = ("ScannedGrain guide (" + draw_url + "scannedgrain.html) and 'Using Practical Grain' (learn.foundry.com/nuke/15.2v1/content/comp_environment/"
                  "color_correction/using_practical_grain.html): 'Frames are selected from a loop of scanned film grain and applied to the input image'; the grain "
                  "plate is made by subtracting the blurred plate and adding 0.5, 'because the ScannedGrain node subtracts 0.5 from the channels when it reads the "
                  "grain file'; offset 'sets the value of neutral gray to subtract from images'; the red, green and blue amount 'increase or decrease on a per-channel "
                  "basis the density of granules ... boosting or reducing the gain of the grain sequence'; saturation 'the intensity of the grain's hue across all "
                  "channels'; low_clip 'a low threshold, based on the input image, below which the grain is not subtracted'. Read as: output = input + (plate - offset) "
                  "* amount per channel, the plate unshifted (seed 0 at frame 0 translates it by nothing). The plate is the fixture C (a colourful plate of values 0 to 1) and "
                  "the picture the fixture A (values from -0.5 to 12), or E for the low clip; the alpha is not compared")
    grain_nb = {"legacy_scannedgrain": 0, "seed": 0, "grain_offset": 0.5, "grain_red": 0.3, "grain_green": 0.4, "grain_blue": 0.5, "grain_saturation": 1.0,
                "grain_low_clip": 0.0}
    grain_knobs = {"offset": 0.5, "amount": "0.3 0.4 0.5", "saturation": 1.0, "low_clip": 0.0, "seed": 0.0}

    def grain_expected(offset=0.5, amount=(0.3, 0.4, 0.5), saturation=1.0, low_clip=0.0, picture="A", only=None):
        def expected(fx):
            src, plate = fx[picture].astype(np.float64), fx["C"].astype(np.float64)
            ref = src.copy()
            valid = np.zeros(src.shape, bool)
            for y in range(H):
                for x in range(W):
                    g = plate[y, x, :3] - offset
                    lum = 0.2126 * g[0] + 0.7152 * g[1] + 0.0722 * g[2]
                    g = lum + saturation * (g - lum)
                    g = g * np.array(amount)
                    below = src[y, x, :3] < low_clip
                    g = np.where(below, np.maximum(g, 0.0), g)
                    ref[y, x, :3] = src[y, x, :3] + g
                    rgb = src[y, x, :3]
                    if only is None or only(rgb):
                        valid[y, x, :3] = True
            return ref, valid
        return expected

    def grain_case(id, nb_over, knob_over, expected, statement, picture="A", **kw):
        nb = {**grain_nb, **nb_over}
        knobs = {**grain_knobs, **knob_over}
        # The guide page documents offset, amount, saturation, low_clip and seed, but docs/parity/nuke_nodes.json lists only fullGrain for this node, so the case
        # names no Nuke knob (the knobs it sets are in the statement).
        case(id, "ScannedGrain", "ScannedGrain", (), "ScannedGrain", nb, expected, kw.pop("source", "guide"), statement + " (Nuke knobs: " + ", ".join(f"{k} {v}" for k, v in knobs.items()) + "). " + grain_note,
             nuke_inputs=(picture,), nb_inputs=(picture, "C"), **kw)

    grain_case("scannedgrain_adds_plate_minus_offset_times_amount", {}, {}, grain_expected(),
               "ScannedGrain at its defaults (offset 0.5, amount 0.3, 0.4, 0.5): each channel gains (plate - 0.5) times its amount")
    grain_case("scannedgrain_amount_per_channel", {"grain_red": 1.0, "grain_green": 0.5, "grain_blue": 2.0}, {"amount": "1 0.5 2"},
               grain_expected(amount=(1.0, 0.5, 2.0)), "ScannedGrain amount 1, 0.5, 2: the grain of each channel is multiplied by its own amount")
    grain_case("scannedgrain_offset_is_the_neutral_grey", {"grain_offset": 0.2}, {"offset": 0.2}, grain_expected(offset=0.2),
               "ScannedGrain offset 0.2: the neutral grey subtracted from the plate is 0.2, so the grain is (plate - 0.2) times the amounts")
    grain_case("scannedgrain_grain_ignores_the_input_level", {"grain_low_clip": -10.0}, {"low_clip": -10.0}, grain_expected(low_clip=-10.0, picture="L"),
               "ScannedGrain on a picture whose levels run from below black to far above white (low clip out of the way): the grain added is the same at every level",
               picture="L")
    grain_case("scannedgrain_saturation_zero_is_monochrome", {"grain_saturation": 0.0, "grain_red": 0.5, "grain_green": 0.5, "grain_blue": 0.5},
               {"saturation": 0.0, "amount": "0.5 0.5 0.5"}, grain_expected(saturation=0.0, amount=(0.5, 0.5, 0.5)),
               "ScannedGrain saturation 0 with equal amounts: the grain has no hue, every channel gains the same value (the luminance of the plate minus the offset, Rec. 709 weights, times the amount)",
               source="convention")
    grain_case("scannedgrain_low_clip_stops_subtracting", {"grain_low_clip": 0.5}, {"low_clip": 0.5},
               grain_expected(low_clip=0.5, picture="E", only=lambda rgb: bool(np.all(rgb < 0.5) or np.all(rgb >= 0.5))),
               "ScannedGrain low clip 0.5 on the textured fixture E: where the input is below 0.5 the grain is not subtracted (negative grain is dropped); pixels with some "
               "channels each side of 0.5 are not compared", picture="E")
    grain_case("scannedgrain_frame_selection", {}, {}, grain_expected(), "ScannedGrain: which frame of the grain loop a frame and a seed select",
               source="convention", unscored="the guide says frames are 'selected from a loop' of about 50 scanned frames and that 'a different grain pattern is produced for each frame' without saying "
               "which frame a given frame number and seed pick; NodeBased has one plate image and translates it by seed + frame instead (a seed of minus the frame freezes it), so the pattern "
               "cannot be compared with Nuke's without guessing")

    return out
