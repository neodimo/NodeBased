"""The 2D behaviour requirements of November 2026, measured for tools/audit_nuke_fidelity.py.

Three checks the first audit (knob names, defaults, sockets, chrome, formula cases) could not see:

* acts:   every paired knob changes NodeBased's output pixels on the fixture when it is moved off its default
          (DiMo 11/1: a knob name without behaviour is the gap). Probed on `Evaluator._kernel`, no Nuke needed.
* layout: the node's knobs sit in Nuke's order with Nuke's labels (DiMo 11/1 10:36 PM: "match known layouts").
* splits: a Nuke AColor knob (r g b a) or WH knob (w h) is a splittable NodeBased param (core.SPLITTABLE), not one scalar
          (DiMo 11/1 10:36 PM: "how some can hold multiple channel values").

Everything here is deterministic: same checkout, same numbers, so the report's drift test keeps working.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SPLIT_JSON = ROOT / "docs" / "parity" / "nuke_split_knobs.json"
PAIRED = ("same", "renamed", "components")       # pairings that name NodeBased params a probe can move ("panel" ones live elsewhere)
ACT_TOL = 1e-6                                   # a knob acts when some output value moves by more than this
_PROBES = {}
# NodeBased kinds that hand their input on unchanged (taps and wiring). Their behaviour is the input, so neither a knob probe nor
# a missing formula case says anything about them.
TAPS = ("Viewer", "Write", "NoOp", "Dot", "PostageStamp", "Output", "Profile", "PlanarTracker")
# Nuke knobs whose effect a one-frame pixel probe cannot see: they act over time, on metadata or on the bounding box only.
# They are left out of `acts` (listed in the report with this reason), never counted as inert.
NOT_PIXEL = {
    "*": {name: "acts across frames (frame range, retiming, motion blur)" for name in (
              "first", "last", "before", "after", "frame", "frame_mode", "origfirst", "origlast", "origset", "motionblur", "shutter",
              "shutteroffset", "shuttercustomoffset")}
         | {name: "acts on metadata, not pixels" for name in ("metainput", "rangeinput")}
         | {name: "acts on the bounding box only, outside the fixture's format" for name in ("pbb", "intersect")},
    # Lane 9 step F (11/2): DropShadow's merge needs the bg input; DirBlur's centre, sample count and pixel offset belong to zoom and radial blur, the default type being linear.
    # Lane 9 step L (11/3): MotionBlur's shutter knobs act on the motion between frames, which the one-frame fixture lacks; with FgVecs wired they act
    # (tests/test_2d_behaviour_filter_l.py). The Denoise and VectorBlur knobs are probed on the fixture.
    "MotionBlur": {"shutterSamples": "the points along the motion vector; a still fixture has no motion, and the probe wires no FgVecs (tests/test_2d_behaviour_filter_l.py streaks a plate moving a whole number of pixels)",
                   "shutterTime": "the exposure as a share of the motion between frames; a still fixture has none, and the probe wires no FgVecs (tests/test_2d_behaviour_filter_l.py)"},
    # Lane 9 step P: viewer controls and undocumented rendering switches have no documented image-pixel behavior.
    "GridWarp": {"source_grid_visible": "viewer overlay only; does not change warped pixels",
                 "source_grid_locked": "viewer editing lock only; does not change warped pixels",
                 "destination_grid_visible": "viewer overlay only; does not change warped pixels",
                 "destination_grid_locked": "viewer editing lock only; does not change warped pixels",
                 "source_color": "colors the source-grid viewer overlay only",
                 "destination_color": "colors the destination-grid viewer overlay only",
                 "source_grid_transform_matrix": "the guide names an extra matrix control but gives no matrix interpretation or pixel behavior",
                 "destination_grid_transform_matrix": "the guide names an extra matrix control but gives no matrix interpretation or pixel behavior"},
    "SplineWarp": {"previewrez": "preview resolution changes editor preview quality only; the guide gives no rendered-pixel behavior",
                   "boundary_curverez": "the guide lists boundary-curve resolution but gives no rendered-pixel behavior; NodeBased has no separate boundary curve payload",
                   "rbf": "the guide names Classic warping but specifies no pixel formula or direction for the switch",
                   "transform_matrix": "the guide names an extra matrix control but gives no matrix interpretation or pixel behavior"},
    # Tracker's zoom and keyframe strips are viewer presentation controls. The Tracker guide
    # describes their display, and they cannot alter the analyzed image pixels.
    "Tracker": {name: "viewer display only (Nuke 17.0 Tracker reference guide)" for name in (
        "zoom_window_behavior", "zoom_window_size", "zoom_magnification_size", "zoom_window_filter_behaviour",
        "keyframe_display", "keyframe_size", "max_number_of_keyframe_tracks_to_display")},
    # Lane 9 step Y (11/4): these controls are exposed/mapped, but their formulas are absent or their behavior is descriptive only.
    "LensDistortion": {
        "distortionModelDisplayX": "equation display only (Nuke 17.0 LensDistortion reference guide)",
        "distortionModelDisplayY": "equation display only (Nuke 17.0 LensDistortion reference guide)",
        "distortionModelType": "the guide names Radial Asymmetric and Radial-Tangential variants without equations; only Radial Standard is specified",
        "distortionOrder": "the guide describes polynomial order but gives no general coefficient-series formula",
        "distortionExponent": "the guide describes polynomial exponents but gives no general coefficient-series formula",
        "normalisationType": "the guide names normalization modes but gives no conversion factors",
        "projection": "the guide lists fisheye projection families but gives no projection equations",
        "focal": "projection parameter has no documented pixel-mapping equation in the guide",
        "sensorSize": "projection parameter has no documented pixel-mapping equation in the guide",
        "distortInFisheyeSpace": "the guide describes operation order but does not define the projection conversion",
        "useFisheye": "the guide describes operation order but does not define the projection conversion",
        "enableBeamSplitter": "the guide describes beam-splitter controls without a pixel equation",
        "beamSplitterDirection": "the guide describes beam-splitter controls without a pixel equation",
        "beamSplitterBending": "the guide describes beam-splitter controls without a pixel equation",
    },
    "Flare": {"chroma_shift_angle_add": "adds the image-centre-to-Position angle only when the flare is off-centre and chroma shift is non-zero; verified by FlareTests, while the one-knob audit fixture is centred",
              "vis_in": "the audit fixture places Position at image centre; visibility range only matters when Position is displaced (tests/test_2d_behaviour_extra_draw.py)",
              "vis_out": "the audit fixture places Position at image centre; visibility range only matters when Position is displaced (tests/test_2d_behaviour_extra_draw.py)",
              "MultiPenta": "preset button writes a collection of other controls; the guide gives no values for its preset (Flare reference guide, Presets tab)",
              "LgRainbow": "preset button writes a collection of other controls; the guide gives no values for its preset (Flare reference guide, Presets tab)",
              "Bright": "preset button writes a collection of other controls; the guide gives no values for its preset (Flare reference guide, Presets tab)",
              "ManyBright": "preset button writes a collection of other controls; the guide gives no values for its preset (Flare reference guide, Presets tab)",
              "Symmetry4": "preset button writes a collection of other controls; the guide gives no values for its preset (Flare reference guide, Presets tab)",
              "all": "reset button clears presets and restores controls; it is an action, not stored rendered state (Flare reference guide, Presets tab)"},
    "Sparkles": {"spark_falloff": "only affects Sparkles type; the audit probe starts in Rays mode (tests/test_2d_behaviour_extra_draw.py verifies the knob in Sparkles mode)",
                 "broken_affected": "only acts when Broken rays is enabled (tests/test_2d_behaviour_extra_draw.py)",
                 "broken_start": "only acts when Broken rays is enabled (tests/test_2d_behaviour_extra_draw.py)",
                 "random_broken_start": "only acts when Broken rays is enabled (tests/test_2d_behaviour_extra_draw.py)",
                 "broken_holes": "only acts when Broken rays is enabled (tests/test_2d_behaviour_extra_draw.py)",
                 "random_broken_holes": "only acts when Broken rays is enabled (tests/test_2d_behaviour_extra_draw.py)",
                 "sparks_direction": "only acts when Sparks direction clipping is enabled (tests/test_2d_behaviour_extra_draw.py)",
                 "sparks_angle": "only acts when Sparks direction clipping is enabled (tests/test_2d_behaviour_extra_draw.py)"},
    "DropShadow": {"operation": "merges the shadowed picture over the bg input, so it acts only with bg wired (tests/test_2d_behaviour_filter_f.py)"},
    "DirBlur": {"BlurCenter": "the origin of a zoom or radial blur (a linear blur uses it only with UseTarget on); the default type is linear (tests/test_2d_behaviour_filter_f.py)",
                "Samples": "the number of taps of a zoom or radial blur; the default type is linear (tests/test_2d_behaviour_filter_f.py)",
                "pixeloffset": "offsets the samples of a zoom blur; the default type is linear (tests/test_2d_behaviour_filter_f.py)",
                "holdout": "keeps the held-out pixels out of a zoom blur's average, and the picture's alpha is the holdout channel; the default type is linear (tests/test_2d_behaviour_filter_k.py)"},
    # Lane 9 step K (11/3): a quality knob that scales the picture down once the filter is wider than it acts only at a size past it, and Laplacian's crop
    # to format acts on the bounding box only; Dither's static seed acts across frames.
    "Laplacian": {"quailty": "caps the filter's size and so acts only once the size passes it; the default size never does (tests/test_2d_behaviour_filter_k.py moves it at size 60)",
                  "crop": "crop to format acts on the bounding box only"},
    "Emboss": {"blurquality": "caps the pre-blur's size and so acts only once the blur size passes it; the default blur size 0 never does (tests/test_2d_behaviour_finish_o.py moves it at size 12)"},
    "EdgeDetect": {"blurquality": "caps the pre-blur's size and so acts only once the blur size passes it; the default blur size 0 never does (tests/test_2d_behaviour_filter_k.py moves it at size 60)"},
    "Dither": {"static_seed": "holds the seed across frames, so a one-frame probe cannot see it (tests/test_2d_behaviour_draw.py compares two frames)"},
    "LightWrap": {"shadows_gain": "grades only the darkest part of the wrap; the fixture's wrap has none (tests/test_2d_behaviour_filter_f.py moves it on a dark wrap)"},
    "Blur": {"crop": "crop to format acts on the bounding box only"},
    # Lane 9 step O (11/3): Defocus's two methods give the same picture (the guide puts the difference below 0.00001); EdgeExtend's edge mask is on a path the probe does reach.
    "Defocus": {"method": "accelerated and fullprecision are two ways to the same picture (the guide puts the difference below 0.00001); NodeBased has one algorithm (tests/test_2d_behaviour_edge_defocus_o.py)"},
    "Sharpen": {"crop": "crop to format acts on the bounding box only",
                "quailty": "caps the filter's size and so acts only once the size passes it; the default size 1 never does (tests/test_2d_behaviour_filter_g.py moves it at size 40)"},
    "Soften": {"crop": "crop to format acts on the bounding box only",
               "quailty": "caps the filter's size and so acts only once the size passes it; the default size never does (tests/test_2d_behaviour_filter_k.py moves it at size 60)"},
    "Glow": {"crop": "crop to format acts on the bounding box only"},
    # Lane 9 step G (11/3): the To pins are the baked pins, read only while Use baked pins is on (tests/test_2d_behaviour_filter_g.py).
    "VectorCornerPin": {f"to{i}": "the baked pin at the current frame, read only while Use baked pins is on (the Bake Corners button sets it); tests/test_2d_behaviour_filter_g.py"
                        for i in (1, 2, 3, 4)},
    "Transform": {"skew_order": "orders the two skews, so it acts only when skew X and skew Y are both set; the probe moves one at a time (tests/test_2d_behaviour_transform.py)"},
    # Lane 2 step D (11/2): knobs that act only in a state the one-knob probe does not reach; tests/test_2d_behaviour_knobs_color.py shows each acting in that state.
    "Shuffle": {k: "routes a Raster or a layer (the other input, a second output layer), which the array kernel the probe calls never sees; shown on the graph" for k in ("fromInput1", "fromInput2", "out2")},
    "ColorLookup": {k: "feeds the Set RGB, Set RGBA and Set A buttons, which add curve points; the kernel reads the curves, not these two" for k in ("source", "target")},
    "HSVTool": {k: "acts only with Color Replacement on (source and destination colour)" for k in ("srccolor", "dstcolor")}
    | {k: "acts only with a range narrower than the whole and an adjustment to roll off, which is two other knobs moved" for k in ("huerolloffrange", "satrolloffrange", "brtrolloffrange")}
    | {k: "inverts the range's mask, which acts only once the matching mask channel (or socket) is chosen and an adjustment is made (tests/test_2d_behaviour_hsv_vf_p.py)" for k in ("invert_hue_mask", "invert_sat_mask", "invert_bright_mask")},
    "SoftClip": {"softclip_min": "acts only with a conversion chosen; the default conversion is none"},
    "Toe": {"invert": "inverts the mask channel, so it acts only with the mask channel on"},
    "KeyMix": {"invertMask": "inverts the mask channel, so it acts only with a mask wired"},
    "ChromaKeyer": {k: "acts only with a replace mode other than none" for k in ("_replaceAlphaDiff", "_replaceColor", "_replaceAmount")},
    "Keylight": {"despillBias": "acts only with the despill bias ganged to the alpha bias off",
                 "insideReplaceMethod": "acts only with an inside matte wired", "insideReplaceColour": "acts only with an inside matte wired",
                 "insideComponent": "acts only with an inside matte wired", "outsideComponent": "acts only with an outside matte wired",
                 "SourceXMethod": "acts only with the source cropped, which is another knob moved", "SourceYMethod": "acts only with the source cropped, which is another knob moved"},
    "IBKColor": {"Size": "acts on a plate with a keyed-out patch to fill (screen erode and patch black set, as in tests/test_2d_behaviour_knobs_color.py); the fixture is not such a plate",
                 "mult": "acts on a plate with a keyed-out patch to fill, as above"},
    "GenerateLUT": {name: "shapes the file the Generate button writes and not the picture, which GenerateLUT passes through (tests/test_2d_behaviour_knobs_k.py writes each and reads it back)"
                    for name in ("file_type", "lut1d", "style1d", "bitdepth_id", "output_bitdepth_id", "clamp_output")},
    "MatchGrade": {"analysisType": "acts across frames (reference frames or every frame); NodeBased's live match measures the frame it shows either way, and Analyze bakes reference frames (tests/test_2d_behaviour_knobs_k.py)",
                   "applyToMask": "acts only with a mask other than none, which is another knob moved (tests/test_2d_behaviour_knobs_k.py)",
                   # Step K finish (11/3): the reference frames, the fit and the export act in the buttons, not in the kernel the probe calls.
                   **{name: "names the frames Analyze Reference Frames and Recalculate CDL read; the kernel shows the frame it is given (tests/test_2d_behaviour_matchgrade_k2.py bakes a match from several frames)"
                      for name in ("sourceRefFrames", "targetRefFrames")},
                   **{name: "shapes the LUT file Write exports and not the picture (tests/test_2d_behaviour_matchgrade_k2.py writes each and reads it back)"
                      for name in ("prelut", "lutres", "colorspace", "outfile")},
                   **{name: "steers Recalculate CDL, whose result lands in slope, offset, power and saturation (tests/test_2d_behaviour_matchgrade_k2.py)"
                      for name in ("iterations", "lockSlope", "lockOffset", "lockPower", "lockSaturation", "cdlError")}},
    "HueCorrect": {"mix_luminance": "restores the luminance the hue curves changed, so it acts only once a curve has changed it; the default curves are flat (tests/test_2d_behaviour_knobs_k.py moves it on a lum curve)"},
    "IBKGizmo": {"ublu": "acts only with a background wired and luminance matching on",
                 "luma": "acts only with luminance matching and the luminance level's enable box on, which is two other knobs moved (tests/test_2d_behaviour_knobs_k.py)",
                 "ll_enable": "acts only with luminance matching on (tests/test_2d_behaviour_knobs_k.py)",
                 "ubcr": "acts only with a background wired (tests/test_2d_behaviour_knobs_k.py)",
                 "color": "acts only with the screen type pick (tests/test_2d_behaviour_knobs_k.py)"},
    "Merge": {"screen_alpha": "acts only for operations that do not already composite by alpha (plus, minus, multiply and the like), not the default over"},
    # Lane 2 step E (11/2): TimeDissolve's kernel is a Dissolve at one `which`; the frame range and the curve turn the frame into that `which` in the graph walk.
    "TimeDissolve": {k: "acts across frames (the dissolve range and curve turn a frame number into the mix; tests/test_2d_behaviour_merge.py shows the start, middle and end)"
                     for k in ("in", "out", "lut")},
    # Lane 2 step G (11/2): EdgeBlur acts on a hard matte's edge band and on the size/quality relation; the fixture's alpha changes at every pixel.
    "EdgeBlur": {"controlchannel": "picks the channel whose edges are blurred; the fixture's channels all change at every pixel, so the band is full whichever is read (tests/test_2d_behaviour_extra_filter.py shows it on a hard matte)",
                 "edge_mult": "sets the width of the edge band; the fixture's alpha changes at every pixel, so the band is full at every value the probe tries (tests/test_2d_behaviour_extra_filter.py shows the band narrow)",
                 "quality": "acts only when size exceeds quality, which is the size knob moved too (tests/test_2d_behaviour_filter_a.py shows the scaling for Blur; the cases here do the same for EdgeBlur)",
                 "crop": "crop to format acts on the bounding box only (tests/test_2d_behaviour_extra_filter.py shows the window grow)"},
    # Lane 2 step P (11/4): the Look transform's reload re-reads look files that NodeBased reads on every render, and ignore errors acts only when the named look is missing.
    "OCIOLookTransform": {"reload": "re-reads the look files, which NodeBased reads on every render (tests/test_2d_behaviour_ocio_p.py)",
                          "ignore_errors": "acts only when the named look is missing from the config; the probe names none (tests/test_2d_behaviour_ocio_p.py)"},
    "OCIOFileTransform": {"interpolation": "tetrahedral and best differ from linear only on a 3D LUT whose corners are coupled, and OpenColorIO renders nearest as linear; the probe's file is a 1D LUT (tests/test_2d_behaviour_ocio_p.py)"},
    "RotoPaint": {
        "hide_progress_bar": "progress display during the tracking action; it does not affect rendered pixels (tests/test_roto_tracking_x.py)",
        "format": "the node inherits the wired image format; a read-only current-format indication cannot change pixels",
        "view": "NodeBased has no stereo views; a view selector cannot change rendered pixels",
        "stereo_offset": "NodeBased has no stereo views; a stereo offset cannot change rendered pixels",
        "source_view": "NodeBased has no stereo views; a source-view selector cannot change rendered pixels",
    },
    "DustBust": {"maskchannel": "writes a named coverage layer; it does not change the RGBA picture (tests/test_2d_behaviour_dustbust_z.py)"},
    "Crop": {"reformat": "makes the box the format, which changes the window and not the fixture's pixels (tests/test_2d_behaviour_transform.py)",
             "crop": "black outside acts only where the box reaches past the picture, with intersect off; the probe's box does not (tests/test_2d_behaviour_transform.py)"},
}
# Nuke nodes whose knobs only act on data the fixture does not hold, so the probe is skipped for the whole node.
NOT_PIXEL["Flare"].update({name: ("multiple-flare controls only act when mult_num is greater than 1; pixel variation cases cover this state (tests/test_2d_behaviour_extra_draw.py)")
                           for name in ("mult_offset", "mult_size", "mult_color", "mult_brightness", "mult_visibility", "mult_flattening", "mult_rotate")})
NOT_PIXEL["Flare"].update({name: ("the per-family seed only changes deterministic randomization when its matching variation is nonzero and mult_num is greater than 1 (tests/test_2d_behaviour_extra_draw.py)")
                           for name in ("mult_seed", "mult_offset_seed", "mult_size_seed", "mult_color_seed", "mult_brightness_seed", "mult_visibility_seed", "mult_flattening_seed", "mult_rotate_seed")})
NOT_PROBED = {
    "Tracker": "tracks are solved across frames; a one-frame probe cannot move them",
    "Roto": "shapes live in the node's payload and the default node holds none",
    "RotoPaint": "strokes live in the node's payload and the default node holds none",
    # Lane 9 step H (11/3): the grids and curves live in the node's payload; a default node holds none, so every knob leaves the picture alone (tests/test_2d_behaviour_warp_h.py moves each on a warp).
    "GridWarp": "the grids live in the node's payload and the default node holds none (tests/test_2d_behaviour_warp_h.py moves each knob on a warp)",
    # Lane 9 step N (11/3): the tracking helpers need data the one-frame fixture lacks.
    "VectorDistort": "the warp needs a SmartVector measured on a sequence, which a one-frame probe lacks; tests/test_2d_behaviour_transform_n.py moves each knob on a plate that moves three whole pixels",
    "PointsTo3D": "the node needs an animated camera and tracked points, which the one-frame fixture lacks; tests/test_2d_behaviour_transform_n.py moves each knob on a known camera and 3D point",
    "Reconcile3D": "the node needs a camera and a 3D point and outputs a track, not pixels; tests/test_2d_behaviour_transform_n.py moves each knob on a known camera and 3D point",
    "SplineWarp": "the curve pairs live in the node's payload and the default node holds none (tests/test_2d_behaviour_warp_h.py moves each knob on a warp)",
}


def not_pixel_reason(nuke_key, knob_name):
    return NOT_PIXEL.get(nuke_key, {}).get(knob_name) or NOT_PIXEL["*"].get(knob_name)


def load_split_table(path=SPLIT_JSON):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------------------------- acts
def _inputs_for(kind, fx, size):
    from nodebased import core
    spec = core.SPECS[kind]
    images = [fx["A"], fx["B"], fx["C"]]
    slots = [images[i % len(images)] for i in range(len(spec["inputs"]))]
    slots += [None] * len(spec.get("optional_inputs", []))     # no mask, no optional input: the node's own behaviour
    return slots


# Starting values the probe uses in place of a kind's defaults where the defaults put the effect outside the 16 x 12 fixture. Nuke's own defaults stay
# the defaults the `defaults` component compares; only the probe's starting point moves.
PROBE_BASE = {
    "MarkerRemoval": {"mode": "bilinear", "linearsearch": 1, "ctrlMask": "fixture.control"},
    "CopyRectangle": {"area_x": 3.0, "area_y": 2.0, "area_r": 11.0, "area_t": 9.0},     # Nuke's 512 389 1536 1167 lies off a 16 x 12 picture
    # Lane 2 step H (11/3): Flare's centre and radii, Glint's 50 pixel rays and Sparkles's 300 pixel rays all lie off a 16 x 12 picture.
    "Flare": {"position_x": 8.0, "position_y": 6.0, "flare_radius_inner": 2.0, "flare_radius_middle": 4.0, "flare_radius_outer": 7.0},
    "Glint": {"length": 6.0, "glint_steps": 3, "tolerance": 0.3},
    "Sparkles": {"spark_x": 8.0, "spark_y": 6.0, "spark_size": 7.0, "spark_ray_nb": 6},
}


# The OCIO nodes are probed against the small config of tools/nuke_fidelity_ocio_fixture.py (the built-in ACES config has hundreds of spaces and no
# fixed answer): the probe starts from a plain pass-through of that config (or, for a look and a file, from the look and the file named) and moves each
# space, look, display, view, file and direction to another of the config's.
_OCIO_FILES = []


def _ocio_files():
    if not _OCIO_FILES:
        import tempfile
        import nuke_fidelity_ocio_fixture as fixture
        _OCIO_FILES.append(fixture.write(tempfile.mkdtemp(prefix="nb-ocio-probe-")))
    return _OCIO_FILES[0]


def probe_base(kind):
    """The probe's starting values for `kind`: PROBE_BASE, and for the OCIO nodes the fixture config and a pass-through request."""
    if kind not in OCIO_PROBE:
        return dict(PROBE_BASE.get(kind, {}))
    config, cube = _ocio_files()
    return {"config": config, **{k: (cube if v == "$cube" else v) for k, v in OCIO_PROBE[kind][0].items()}}


OCIO_PROBE = {
    "OCIOColorspace": ({"src": "lin", "dst": "lin"}, {"src": ["gain2"], "dst": ["gain2"], "transform_direction": ["inverse"]}),
    "OCIOLookTransform": ({"src": "lin", "dst": "lin", "look": "double"}, {"src": ["gain2"], "dst": ["gain2"], "look": [""], "transform_direction": ["inverse"]}),
    "OCIODisplay": ({"display": "Mon", "view": "Plain"}, {"src": ["gain2"], "display": ["Alt"], "view": ["Bright"], "transform_direction": ["inverse"]}),
    "OCIOFileTransform": ({"path": "$cube"}, {"path": [""], "transform_direction": ["inverse"]}),
    "OCIOLogConvert": ({}, {}),
}


def probe_moves(kind, name, value):
    """Values to move a string or choice param of an OCIO node to (the config's own spaces and names), else the generic candidates."""
    if kind in OCIO_PROBE and name in OCIO_PROBE[kind][1]:
        config, cube = _ocio_files()
        return [cube if v == "$cube" else v for v in OCIO_PROBE[kind][1][name] if v != value]
    return candidates(name, value)


def _base_params(kind, size):
    from nodebased import core
    p = dict(core.SPECS[kind]["params"], **probe_base(kind))
    h, w = size
    for name, value in (("width", w), ("height", h)):          # generators render at the fixture's size, not 1920 x 1080
        if name in p and isinstance(p[name], (int, float)) and not isinstance(p[name], bool):
            p[name] = value
    return p


def _run(kind, params, slots):
    from nodebased.imaging import Evaluator
    out = Evaluator._kernel(kind, params, slots)
    if isinstance(out, tuple):
        out = out[0]
    arr = np.asarray(out, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] < 1:
        raise TypeError("not an image")
    return arr


def candidates(name, value):
    """Values to move one param to, off its default, inside its LIMITS and CHOICES. Empty when the param is not a scalar we can move."""
    from nodebased import core
    choices = core.CHOICES.get(name)
    if choices is not None:
        return [c for c in choices if c != value][:8]
    if isinstance(value, bool):
        return [not value]
    if not isinstance(value, (int, float)):
        return []
    lo, hi = core.LIMITS.get(name, (None, None))
    raw = [value + 1, value - 1, value * 2 + 0.5, value * 0.5 + 0.25, value + 0.37, value + 7]
    if isinstance(value, int) and lo == 0 and hi == 1:
        raw = [1 - value]
    elif isinstance(value, int):
        raw = [int(round(v)) for v in raw]
    out = []
    for v in raw:
        if lo is not None:
            v = max(lo, v)
        if hi is not None:
            v = min(hi, v)
        if v != value and v not in out:
            out.append(v)
    return out


def _moves(kind, base, slots, ref, name):
    for v in probe_moves(kind, name, base[name]):
        try:
            out = _run(kind, dict(base, **{name: v}), slots)
        except Exception:  # noqa: BLE001 - a value the kernel refuses is not evidence either way
            continue
        if out.shape != ref.shape or np.nanmax(np.abs(np.nan_to_num(out - ref, nan=1.0, posinf=1.0, neginf=1.0))) > ACT_TOL:
            return True
    return False


def probe_acts(kind, params, fx):
    """{param: True | False} for each NodeBased param of `kind` named in `params`, or None when the kernel cannot run on the fixture.

    A param that moves nothing alone is tried again with each other probed param moved first (Grade's `reverse` is
    the identity at default levels), so only a param that changes nothing in any of those contexts reads False."""
    key = (kind, tuple(sorted(params)))
    if key in _PROBES:
        return _PROBES[key]
    if kind == "DustBust":
        from nodebased.core import Dispatcher, empty_document
        from nodebased.imaging import Evaluator
        from nodebased.core import SPECS
        size = fx["A"].shape[:2]; h, w = size
        base = dict(SPECS[kind]["params"], **probe_base(kind))
        d = Dispatcher(empty_document()); d.execute({"op":"time","first":0,"last":2,"current":1})
        d.execute({"op":"create","id":"plate","type":"Constant","params":{"width":w,"height":h,"red":0.2,"green":0.3,"blue":0.4,"alpha":1}})
        d.execute({"op":"set_curve","id":"plate","param":"red","curve":{"interpolation":"linear","keys":[{"frame":0,"value":0.9},{"frame":1,"value":0.2},{"frame":2,"value":0.7}]}})
        d.execute({"op":"create","id":"dust","type":"DustBust","params":base})
        d.execute({"op":"set_dustbust_spots","id":"dust","spots":[{"frame":1,"x":w/2,"y":h/2,"radius":max(2,min(w,h)/3)}]})
        d.execute({"op":"connect","id":"dust","input":"image","source":"plate"})
        def render(dispatcher): return Evaluator().evaluate_raster(dispatcher.document,"dust",frame=1).pixels
        reference=render(d); acts={}
        for name in params:
            if name not in base: continue
            acts[name]=False
            for value in probe_moves(kind,name,base[name]):
                trial=Dispatcher(d.document)
                trial.execute({"op":"set","id":"dust","param":name,"value":value})
                if not np.array_equal(render(trial),reference): acts[name]=True; break
        _PROBES[key]=acts
        return acts
    if kind == "MarkerRemoval":
        from nodebased.imaging import Evaluator
        h, w = fx["A"].shape[:2]
        yy = np.arange(h, dtype=np.float32)[:, None]
        image = np.zeros((h, w, 4), np.float32)
        image[..., :3] = np.broadcast_to(yy[..., None], (h, w, 3))
        image[..., 3] = 1
        control = np.zeros((h, w, 1), np.float32); control[h//2-1:h//2+2, w//2-3:w//2+3] = 1
        base = _base_params(kind, (h, w)); slots = [image]
        run = lambda p: Evaluator._kernel(kind, p, slots, layer_array=lambda name: control if name == "fixture.control" else None)
        reference = run(base); acts = {}
        for name in params:
            acts[name] = False
            for value in probe_moves(kind, name, base[name]):
                if not np.array_equal(run(dict(base, **{name: value})), reference): acts[name] = True; break
        _PROBES[key] = acts
        return acts
    size = fx["A"].shape[:2]
    base = _base_params(kind, size)
    slots = _inputs_for(kind, fx, size)
    try:
        ref = _run(kind, base, slots)
    except Exception:  # noqa: BLE001
        _PROBES[key] = None
        return None
    names = [n for n in params if n in base and probe_moves(kind, n, base[n])]
    acts = {n: _moves(kind, base, slots, ref, n) for n in names}
    for n in [n for n, ok in acts.items() if not ok]:
        for other in names:
            if other == n:
                continue
            ctx = dict(base, **{other: probe_moves(kind, other, base[other])[0]})
            try:
                ctx_ref = _run(kind, ctx, slots)
            except Exception:  # noqa: BLE001
                continue
            if _moves(kind, ctx, slots, ctx_ref, n):
                acts[n] = True
                break
    _PROBES[key] = acts
    return acts


def acts_component(nuke_key, kind, entries, knobs, fx, weight):
    """(fraction, detail) over the counted Nuke knobs paired with movable NodeBased params; None when nothing can be probed."""
    if nuke_key in NOT_PROBED:
        return None, {"acts_note": NOT_PROBED[nuke_key]}
    knobs = [k for k in knobs if not not_pixel_reason(nuke_key, k["name"])]
    by_nuke = {}
    for e in entries:
        if e["kind"] in PAIRED:
            for n in e["nuke"]:
                by_nuke.setdefault(n, []).extend(e["nodebased"])
    wanted = sorted({p for k in knobs for p in by_nuke.get(k["name"], [])})
    acts = probe_acts(kind, wanted, fx)
    if acts is None:
        return None, {"acts_note": "the kernel does not run on the fixture"}
    total = have = 0
    inert, probed = [], 0
    for k in knobs:
        mine = [p for p in by_nuke.get(k["name"], []) if p in acts]
        if not mine:
            continue
        probed += 1
        total += weight[k["primary"]]
        if any(acts[p] for p in mine):
            have += weight[k["primary"]]
        else:
            inert.append(k["name"])
    if not total:
        return None, {}
    return have / total, {"inert_knobs": inert, "acts_counts": (probed - len(inert), probed)}


# -------------------------------------------------------------------------------------------------------------- layout
def _norm_label(text):
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _longest_increasing(seq):
    """Length of the longest run that never goes back. Knobs that share a row (Reformat's box width and height are one w, h pair)
    sit at the same position, and sharing a row is in order."""
    tails = []
    for x in seq:
        lo, hi = 0, len(tails)
        while lo < hi:
            mid = (lo + hi) // 2
            if tails[mid] <= x:
                lo = mid + 1
            else:
                hi = mid
        tails[lo:lo + 1] = [x]
    return len(tails)


def layout_component(kind, entries, knobs):
    """(fraction, detail): half the share of paired knobs already in Nuke's order (longest run kept in order), half the share
    of paired single-param main-tab knobs whose label reads as the guide's label or the knob's scripting name. None when fewer than two paired knobs are laid out."""
    from nodebased.knobs import KnobGroup, knob_layout
    groups = knob_layout(kind)
    if kind == "ViewMetaData":
        # Transient panel widgets are measured here without pretending they are document params.
        groups = (KnobGroup("table", ("metadata_table",), label="metadata viewer"),
                  KnobGroup("string", ("metadata_search",), label="search metadata for"),
                  KnobGroup("enum", ("metadata_filter_mode",), label="within"))
    position, label_of = {}, {}
    for i, g in enumerate(groups):
        for p in g.params:
            position.setdefault(p, i)
            label_of.setdefault(p, g.label)
    first = {}
    for e in entries:
        if e["kind"] in PAIRED or (kind == "ViewMetaData" and e["kind"] == "panel"):
            for n in e["nuke"]:
                first.setdefault(n, e)
    seq, names = [], []
    for k in knobs:
        e = first.get(k["name"])
        if e is None:
            continue
        spots = [position[p] for p in e["nodebased"] if p in position]
        if spots:
            seq.append(min(spots))
            names.append(k["name"])
    if len(seq) < 2:
        return None, {}
    kept = _longest_increasing(seq)
    order = kept / len(seq)
    labels_total = labels_ok = 0
    wrong = []
    # A guide label shared by several knobs is the row's label (Read's "Frame Range" over first, before, last, after), not one
    # knob's, and "n/a" is the guide having none: neither says what one knob should read.
    shared = {}
    for k in knobs:
        shared[_norm_label(k.get("label"))] = shared.get(_norm_label(k.get("label")), 0) + 1
    for k in knobs:
        e = first.get(k["name"])
        theirs = _norm_label(k.get("label"))
        if not k["primary"] or e is None or len(e["nodebased"]) != 1 or len(e["nuke"]) != 1 or theirs in ("", "na") or shared[theirs] > 1:
            continue
        ours = label_of.get(e["nodebased"][0])
        if ours is None:
            continue
        labels_total += 1
        if _norm_label(ours) in (theirs, _norm_label(k["name"])):
            labels_ok += 1
        else:
            wrong.append((k["name"], ours, k["label"]))
    if labels_total:
        frac = 0.5 * order + 0.5 * labels_ok / labels_total
    else:
        frac = order
    return frac, {"layout_order": (kept, len(seq)), "layout_labels": (labels_ok, labels_total), "label_mismatches": wrong}


# -------------------------------------------------------------------------------------------------------------- splits
def splits_component(nuke_key, kind, entries, table):
    """(fraction, detail) over the Nuke knobs the split table lists for this node; None when the table lists none.

    A knob counts 1 when a paired NodeBased param is in core.SPLITTABLE with the same split set, 0.5 when the pairing names at
    least one NodeBased param per channel (Transform's scale_x / scale_y: per-channel values without Nuke's one-slider form), else 0."""
    from nodebased import core
    from nodebased.core import SPLIT_CHANNELS
    listed = table["knobs"].get(nuke_key)
    if not listed:
        return None, {}
    paired = {}
    for e in entries:
        if e["kind"] in PAIRED:
            for n in e["nuke"]:
                paired.setdefault(n, []).extend(p for p in e["nodebased"] if p not in paired.get(n, []))
    got, missing, partial = 0.0, [], []
    for name, channels in listed.items():
        params = paired.get(name, [])
        if not params or kind is None:
            missing.append(name)
        elif any(core.SPLITTABLE.get(kind, {}).get(p) == channels for p in params):
            got += 1
        elif len(params) >= len(SPLIT_CHANNELS[channels]):
            got += 0.5
            partial.append(name)
        else:
            missing.append(name)
    return got / len(listed), {"split_missing": missing, "split_partial": partial,
                               "split_counts": (len(listed) - len(missing) - len(partial), len(listed))}
