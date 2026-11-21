#!/usr/bin/env python3
"""Write the Nuke reference pack: the fixture images, one Python script per fixture case, and the script that renders them.

Each case is a Python file that builds its small graph (Read nodes for the fixtures, the node under test, a 32-bit float
EXR Write) and calls `nuke.execute` on the Write. They run in Nuke's terminal mode (`nuke -t case.py`), which is what
Nuke Non-commercial allows: command-line renders accept only encrypted `.nknc` scripts, the Python API keeps at most 10
nodes reachable per script (so each case holds the handles `nuke.nodes.X(...)` returns and never looks a node up by name),
and rendered output is limited to 1920x1080. The fixtures are tiny.

The first run of Nuke Non-commercial on a machine needs the one-time Foundry sign-in; until then `render_all.sh` reports
every case as failed with the licence message in the case's log.

    python tools/make_nuke_reference_pack.py PACK_DIR              # writes PACK_DIR/{fixtures,scripts,render_all.sh,MANIFEST.json,README.txt}
    NUKE=/path/to/nuke-nc.sh PACK_DIR/render_all.sh RENDER_DIR     # nuke -t on every script, 32-bit float EXRs into RENDER_DIR
    python tools/audit_nuke_fidelity.py --reference-dir RENDER_DIR # score NodeBased against the renders

`compare_case` and `load_render` below are the loader the scorer uses: per channel, the maximum absolute error and the worst pixel.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from nuke_fidelity_cases import CASES, H, W, make_fixtures, run_nodebased  # noqa: E402

FIXTURE_NAMES = frozenset(make_fixtures())

TOLERANCE = dict(atol=1e-4, rtol=1e-4)          # against 32-bit float Nuke output; looser than the formula tests
CHANNELS = "RGBA"


# ------------------------------------------------------------------------------------------------------------- the pack
# The cases name nodes the way the guide does; Nuke 17's Python knows several of them by another class (11/20, read from nuke.plugins()
# and from the node each class creates). Where Nuke 17 keeps an old version under the guide's name (Grain, ZDefocus, LensDistortion,
# IBKGizmo), the class here is the current node the guide describes.
NUKE17_CLASS = {"OFlow": "OFlow2", "CrossTalk": "CCrosstalk", "ScreenKeyer": "Keylight", "IBKColor": "IBKColourV3",
                "DirBlur2": "DirBlurWrapper", "Exposure": "EXPTool", "Denoise": "Denoise2", "Toe": "Toe2", "AdjustBBox": "AdjBBox",
                "Stabilize": "Stabilize2D", "TVIScale": "TVIscale2", "Backdrop": "BackdropNode", "Grain": "Grain2",
                "ZDefocus": "ZDefocus2", "LensDistortion": "LensDistortion2", "IBKGizmo": "IBKGizmoV3", "Shuffle": "Shuffle2"}
# The guide's Shuffle is Nuke 17's toolbar Shuffle, class Shuffle2 (two inputs, a mappings knob); nuke.nodes.Shuffle() makes the legacy
# one-input node. ShuffleCopy keeps its legacy class, which the guide still describes. Neither has knobs a case can list one by one, so
# the pack writes their routing from NodeBased's parameters (ROUTING_WRITERS below; read in Nuke 17.0v6 on 11/20).
# Knob names the cases spell differently from Nuke 17, keyed by the case's class. A name with "#2" is the guide's second knob of that name.
NUKE17_KNOB = {("Bilateral", "size"): "Size", ("Bilateral", "colourSigma"): "ColorSigma", ("Bilateral", "positionalSigma"): "PositionalSigma",
               ("Colorspace", "illuninant_in"): "illuminant_in", ("Colorspace", "illuninant_out"): "illuminant_out",
               ("Saturation", "luminance_math"): "mode", ("TimeEcho", "framesfade"): "frmaesfade",   # Nuke's own spelling
               ("ScreenKeyer", "screenColor"): "screenColour", ("OFlow", "method"): "interpolation", ("Vectorfield", "file"): "vfield_file",
               ("ZDefocus", "aspect#2"): "aspect", ("VectorCornerPin", "to1#2"): "from1", ("VectorCornerPin", "to2#2"): "from2",
               ("VectorCornerPin", "to3#2"): "from3", ("VectorCornerPin", "to4#2"): "from4", ("LightWrap", "size"): "Diffuse", ("Grain", "invert_mask"): "maskinvert"}
# Enumeration values the cases spell differently from Nuke 17's menu, keyed by Nuke class and knob. Case and spaces are matched anyway.
NUKE17_ENUM = {("Colorspace", "colorspace_in"): {"Gamma1.8": "1.80", "Gamma2.2": "2.20", "Gamma2.4": "2.40", "Gamma2.6": "2.60"},
               ("Colorspace", "colorspace_out"): {"Gamma1.8": "1.80", "Gamma2.2": "2.20", "Gamma2.4": "2.40", "Gamma2.6": "2.60"},
               # Nuke's Colorspace keeps the white apart (illuminant_in/out, D65 by default): P3-D65 is the DCI-P3 primaries on a D65 white.
               ("Colorspace", "primary_in"): {"P3-D65": "DCI-P3"}, ("Colorspace", "primary_out"): {"P3-D65": "DCI-P3"},
               # NodeBased's D60 is the ACES white (0.32168, 0.33767), Nuke's ACES illuminant; its DCI white (0.314, 0.351) is Nuke's DCI-P3.
               ("Colorspace", "illuminant_in"): {"D60": "ACES", "DCI": "DCI-P3"}, ("Colorspace", "illuminant_out"): {"D60": "ACES", "DCI": "DCI-P3"},
               ("Bokeh", "outputType"): {"Focus Mattes": "Focus Matte"},
               ("Vectorfield", "interpolation"): {"trilinear": "trilinear field"}}
NOT_IN_NC = {"Primatte", "Ultimatte", "GenerateLUT"}            # Nuke Non-commercial refuses to create these nodes
# Nodes whose input the pack's one-frame fixtures cannot stand in for (read from the render log of 11/20).
NEEDS_SEQUENCE = {"VectorDistort": "VectorDistort reads a SmartVector input over a frame range; the pack's vector fixture is one frame",
                  "Root": "the project settings are nuke.root(), not a node a script can create and wire (the case is reported, not scored)"}
STRING_KINDS = {"String_Knob", "File_Knob", "EvalString_Knob", "Multiline_Eval_String_Knob", "Channel_Knob", "ChannelMask_Knob",
                "Input_Channel_Knob", "Pulldown_Knob", "Format_Knob", "Font_Knob", "Cascading_Enumeration_Knob",
                "OneView_Knob", "MultiView_Knob", "ViewView_Knob", "CascadingEnumeration_Knob", "EditableEnumeration_Knob",
                "Radio_Knob", "FreeType_Knob"}
INT_KINDS = {"Int_Knob", "Frame_Knob"}
SKIPPED_SUFFIXES = ("_not_a_pixel_behaviour", "_not_in_nodebased")
# Curves on the node under test in the Time cases, keyed by NodeBased's parameter: the Nuke knob they animate.
CURVE_KNOB = {("TimeWarp", "lookup"): "lookup", ("Kronos", "frame"): "timingFrame2", ("OFlow", "frame"): "timingFrame2"}
CURVE_INTERPOLATION = {"linear": "nuke.LINEAR", "constant": "nuke.CONSTANT", "smooth": "nuke.SMOOTH", "cubic": "nuke.CUBIC",
                       "catmull_rom": "nuke.CATMULL_ROM"}
# Frame-range knobs Nuke's GUI fills from the input's range when a node is created on it; a script's nuke.nodes.X() keeps the
# defaults (1 to 100) instead, read on 11/20. The scripts set them the way the GUI would, before the case's own knobs.
GUI_INPUT_RANGE = {"Retime": ("input.first", "input.last", "output.first", "output.last"), "TimeClip": ("first", "last", "origfirst", "origlast"),
                   "Kronos": ("input.first", "input.last"), "OFlow2": ("input.first", "input.last")}
STAMP_BLUE = 0.5                                               # tools/nuke_fidelity_cases.py: the stamped plate's green 0.25 and blue 0.5
# OCIO nodes whose case names a config file of its own (the fixture config in tools/nuke_fidelity_cases.py): the script loads it as Nuke's
# custom OCIO config (root colorManagement OCIO, OCIO_config custom, read in Nuke 17.0v6 on 11/21), and these menu knobs take the config's
# names as text, since the knob table holds the menus of Nuke's default config.
OCIO_MENU_KNOBS = {"in_colorspace", "out_colorspace", "colorspace", "display", "view", "look"}
# A moving plate, an upstream Transform chain or a generator on an extra input is rendered by NodeBased into an EXR sequence beside the
# fixtures and read back in Nuke, so the render tests the node under test on the very frames NodeBased shows it. PLATE_MARGIN frames are
# written on each side of the project range for the nodes that look past it (Kronos and OFlow read neighbouring frames).
PLATE_MARGIN = 3


def nuke17_class(case):
    return NUKE17_CLASS.get(case.nuke_class, case.nuke_class)


def nuke_classes_used():
    """Every Nuke class the pack's scripts create: the nodes under test and the helpers (Read, Expression, FrameRange, Crop, Write...)."""
    return sorted({nuke17_class(c) for c in CASES} | {"Read", "Expression", "FrameRange", "AppendClip", "Crop", "Write"})


def _knob_table():
    from nuke17_knob_table import load_table
    return load_table()


def _norm(text):
    return "".join(ch for ch in str(text).lower() if ch.isalnum() or ch == ".")


def enum_value(cls, knob, value, values):
    """The menu entry of an Enumeration knob for a case's value, or None when Nuke's menu has nothing that matches."""
    value = NUKE17_ENUM.get((cls, knob), {}).get(value, value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return values[int(value)] if 0 <= int(value) < len(values) else None
    if value in values:
        return value
    matches = [v for v in values if _norm(v) == _norm(value)]
    return matches[0] if len(matches) == 1 else None


# NodeBased names the mask input's channels mask.<c>; in Nuke a Read fixture on the mask input carries them as rgba.<c>.
CHANNEL_VALUE = {"mask.a": "rgba.alpha", "mask.r": "rgba.red", "mask.g": "rgba.green", "mask.b": "rgba.blue"}


def case_files(case):
    """{file name: path} for the files a case made for itself (a .cube LUT in a temporary folder), found in its NodeBased params. An
    OCIO config brings the files beside it (its LUTs), which its transforms name relative to the config's folder."""
    found = {}

    def walk(v):
        if isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, str) and len(v) < 4096 and os.path.isabs(v) and os.path.isfile(v):
            found[os.path.basename(v)] = v
            if v.endswith(".ocio"):
                folder = os.path.dirname(v)
                for name in sorted(os.listdir(folder)):
                    if os.path.isfile(os.path.join(folder, name)):
                        found.setdefault(name, os.path.join(folder, name))
    walk(case.nb_params or {})
    return found


def ocio_config(case):
    """The file name of the OCIO config an OCIO case brings (copied beside the fixtures), or None."""
    config = (case.nb_params or {}).get("config")
    if case.nuke_class.startswith("OCIO") and isinstance(config, str) and config.endswith(".ocio") and os.path.isfile(config):
        return os.path.basename(config)
    return None


def case_knob_statement(case, knob, value, table):
    """knob_statement for one of the case's knobs, with the case's own names: the Nuke knob name, its files, and an OCIO config's menus."""
    cls = nuke17_class(case)
    knob = NUKE17_KNOB.get((case.nuke_class, knob), knob)
    if ocio_config(case) and knob in OCIO_MENU_KNOBS:
        if knob not in table.get(cls, {}).get("knobs", {}):
            raise ValueError(f"Nuke 17's {cls} has no knob {knob}")
        return f"node[{knob!r}].setValue({str(value)!r})  # a name in the case's own OCIO config"
    return knob_statement(cls, knob, value, table, case_files(case))


# NodeBased's filter names where Nuke 17's filter menu says it another way; the rest match by name (enum_value ignores case).
NUKE17_FILTER = {"nearest": "impulse", "bilinear": "linear", "box": "impulse"}


def implied_knobs(case, table):
    """Knobs a case sets in NodeBased and leaves off its Nuke list, which the script writes so Nuke does not fall back to its default:
    the resampling filter (11/21: 52 Transform, Reformat, CornerPin2D, LensDistortion, IDistort, STMap and Tile cases ran NodeBased's
    nearest or bilinear against Nuke's cubic)."""
    listed = {k for k, _ in case.nuke_knobs}
    p = case.nb_params or {}
    knobs = table.get(nuke17_class(case), {}).get("knobs", {})
    out = []
    if "filter" in knobs and "filter" not in listed and isinstance(p.get("filter"), str):
        out.append(("filter", NUKE17_FILTER.get(p["filter"], p["filter"])))
    # CornerPin2D's from corners: the GUI fills them from the input's format on creation, a script's node keeps 0 0 on all four (11/21:
    # every CornerPin case rendered black), so the script writes NodeBased's own, which number the corners the way its to corners do.
    for n in range(1, 5):
        if case.nuke_class == "CornerPin2D" and f"from{n}" in knobs and f"from{n}" not in listed and f"from{n}_x" in p:
            out.append((f"from{n}", [float(p[f"from{n}_x"]), float(p[f"from{n}_y"])]))
    return out


# Nuke 17's input order where a case lists its pictures in another: STMap's input 0 is the source and 1 the map (11/21, probed: the
# identity map on input 1 returns the source; the cases list the map first).
NUKE17_INPUT_ORDER = {"STMap": (1, 0)}
VECTOR_LAYER = "nbv"


def vectors_on_the_source(case):
    """True for an IDistort case that lists only its vector picture: Nuke's IDistort reads the vectors from channels of its one input,
    so the script copies the vector fixture's four channels into a layer of their own (nbv) on the source and points uv at that layer
    (probed in Nuke 17.0v6 on 11/21: the guide's u -1 v 5 then reads the pixel one left and five up)."""
    return case.nuke_class == "IDistort" and len(case.nuke_inputs) == 1 and len(case.nb_inputs) == 2 and case.nb_inputs[1] == case.nuke_inputs[0]


def fractional_frame(case):
    """The case's frame when it is fractional, else None. Nuke's terminal mode renders whole frames, so the script renders frame 1 through
    a TimeWarp after the node under test whose lookup is the fractional frame and whose filter is none: it asks the node for exactly that
    frame (probed in Nuke 17.0v6 on 11/21: lookup 0.5 on a frame-stamped plate reads 0.5, and NoTimeBlur under lookup 2.4 rounds as
    NodeBased's does)."""
    stamp = _stamp(case)
    if stamp is None:
        return None
    frame = float(stamp.get("frame", 1))
    return frame if frame != int(frame) else None


def sequence_plate(case):
    """True when the case's picture comes from NodeBased as an EXR sequence: a moving plate, or an upstream chain with more than FrameRange."""
    stamp = _stamp(case)
    if stamp is None or "clips" in stamp:
        return False
    return "plate" in stamp or any(step[0] != "FrameRange" for step in stamp.get("chain", ()))


def plate_range(case):
    stamp = _stamp(case)
    first, last = stamp.get("time") or (1, max(1, int(float(stamp.get("frame", 1)))))
    return int(first) - PLATE_MARGIN, int(last) + PLATE_MARGIN


def plate_sequences(case):
    """{file name: (H, W, 4) array} of the pictures the case's node reads, made by NodeBased: the plate (after its chain) on every frame of
    plate_range as `<case>_plate.<frame>.exr`, and each extra input's generator once as `<case>_<slot>.exr`. The plate is the case's chain
    run with its last step as the node shown, so it is built by the same document code the case's own answer is."""
    from dataclasses import replace
    from nuke_fidelity_cases import _run_stamped_document
    stamp = dict(_stamp(case))
    chain = list(stamp.get("chain", ()))
    out = {}
    if chain:
        kind, params, *curves = chain[-1]
        base = dict(stamp, chain=chain[:-1], curves=curves[0] if curves else {})
    else:                                        # the plate alone: a pass-through Dot-like step NodeBased has (FrameRange over the range)
        kind, params = "FrameRange", {"first_frame": -10000, "last_frame": 10000}
        base = dict(stamp, chain=[], curves={})
    base.pop("inputs", None)
    first, last = plate_range(case)
    for frame in range(first, last + 1):
        step = replace(case, nb_kind=kind, nb_params=dict(params, _stamp=dict(base, frame=frame)))
        out[f"{case.id}_plate.{frame}.exr"] = np.asarray(_run_stamped_document(step), np.float32)
    if stamp.get("inputs"):
        from nodebased.core import Dispatcher, empty_document
        from nodebased.imaging import Evaluator
        width, height = stamp.get("size", (W, H))
        for slot, (gen, gparams) in stamp["inputs"].items():
            d = Dispatcher(empty_document())
            d.execute({"op": "create", "id": "g", "type": gen, "params": {"width": width, "height": height, **gparams}})
            raster = Evaluator().evaluate_raster(dict(d.document, view="g"), target="g", frame=1)
            out[f"{case.id}_{slot}.exr"] = np.asarray(raster.pixels, np.float32)
    return out


def _number_list(text):
    """The numbers of a Tcl-style list of plain numbers ('{3 2}', '{{0.5 1 2} {0 1 0}}'), flattened, or None for anything else."""
    try:
        return [float(t) for t in text.replace("{", " ").replace("}", " ").split()] or None
    except ValueError:
        return None


def knob_statement(cls, knob, value, table, files=()):
    """The Python line that sets one knob of the node under test, written for the knob's kind in Nuke 17 (the table): an enumeration by
    its menu index, an integer knob with an int, a text knob with the text, and a number, colour or XY knob given as text ("4 6", "12",
    a Tcl list or an expression) through fromScript, which is how Nuke reads a knob from text. Raises ValueError with the reason when
    Nuke has no such knob or menu entry."""
    knobs = table.get(cls, {}).get("knobs")
    if knobs is None:
        raise ValueError(f"Nuke 17 has no node class {cls}" + (f" ({table[cls]['error']})" if cls in table and "error" in table[cls] else ""))
    entry = knobs.get(knob)
    if entry is None:
        raise ValueError(f"Nuke 17's {cls} has no knob {knob}")
    kind, values = (entry[0], entry[1]) if isinstance(entry, list) else (entry, [])
    if kind == "Enumeration_Knob" and values:
        if isinstance(value, bool):
            value = int(value)
        chosen = enum_value(cls, knob, value, values)
        if chosen is None:
            raise ValueError(f"Nuke 17's {cls}.{knob} has no menu entry {value!r} (it offers {', '.join(values[:12])})")
        return f"node[{knob!r}].setValue({values.index(chosen)})  # {chosen}"
    if kind == "Boolean_Knob":
        on = value if isinstance(value, bool) else str(value).strip().lower() in ("1", "1.0", "true", "on", "yes")
        return f"node[{knob!r}].setValue({bool(on)!r})"
    if kind == "File_Knob" and isinstance(value, str) and value and not os.path.isabs(value):
        if value not in files:
            raise ValueError(f"the case's file {value} is not one the pack can write (NodeBased's case makes it under another name)")
        return f"node[{knob!r}].setValue(os.path.join(fixtures, {value!r}).replace(os.sep, '/'))"
    if kind == "Format_Knob" and isinstance(value, str) and len(value.split()) == 2 and all(v.isdigit() for v in value.split()):
        w, h = value.split()
        return f"nuke.addFormat('{w} {h} 1 nb_{w}x{h}')\nnode[{knob!r}].setValue('nb_{w}x{h}')"
    if kind in ("Channel_Knob", "ChannelMask_Knob", "Input_Channel_Knob") and value in CHANNEL_VALUE:
        value = CHANNEL_VALUE[value]
    if kind in INT_KINDS and not isinstance(value, str):
        return f"node[{knob!r}].setValue({int(round(float(value)))!r})"
    if kind in STRING_KINDS or kind == "Enumeration_Knob":
        return f"node[{knob!r}].setValue({str(value)!r})"
    if isinstance(value, str):
        numbers = _number_list(value)
        if numbers is not None and "{" in value:   # fromScript('{3 2}') on an XY knob sets only x (11/20), so a braced list of numbers goes in as numbers
            return f"node[{knob!r}].setValue({numbers[0]!r})" if len(numbers) == 1 else f"node[{knob!r}].setValue({numbers!r})"
        return f"node[{knob!r}].fromScript({value!r})"
    if isinstance(value, (list, tuple)):
        return f"node[{knob!r}].setValue({[float(v) for v in value]!r})"
    return f"node[{knob!r}].setValue({float(value)!r})"


_SHUFFLE_CHANNEL = {"R": "rgba.red", "G": "rgba.green", "B": "rgba.blue", "A": "rgba.alpha"}
_SHUFFLE_OUT = (("red_from", "rgba.red"), ("green_from", "rgba.green"), ("blue_from", "rgba.blue"), ("alpha_from", "rgba.alpha"))


def shuffle2_lines(case):
    """NodeBased's Shuffle routing as Nuke 17's Shuffle2: in1 reads input 0 and, with a second input, in2 reads input 1 (the 'A' entry
    of fromInput2, whose menu a script only gets by setting it). Each mapping is (slot, source, destination); slot -1 with 'black' or
    'white' is a constant. NodeBased's '0' and '1' are black and white, 'R2' is the second input's red."""
    p = case.nb_params or {}
    two = len(case.nuke_inputs) > 1
    mappings = []
    for param, dest in _SHUFFLE_OUT:
        route = str(p.get(param, param[0].upper()))
        if route in ("0", "1"):
            mappings.append((-1, "black" if route == "0" else "white", dest))
        elif route.endswith("2") and route[0] in _SHUFFLE_CHANNEL:
            mappings.append((1, _SHUFFLE_CHANNEL[route[0]], dest))
        elif route in _SHUFFLE_CHANNEL:
            mappings.append((0, _SHUFFLE_CHANNEL[route], dest))
        else:
            raise ValueError(f"NodeBased's Shuffle route {route!r} has no Shuffle2 mapping in the pack")
    lines = ["node['in1'].setValue('rgba')"]
    if two:
        lines += ["node['in2'].setValue('rgba')", "node['fromInput2'].setValues(['B', 'A'])", "node['fromInput2'].setValue(1)  # A: input 1"]
    lines.append(f"node['mappings'].setValue({mappings!r})")
    return lines


def shufflecopy_lines(case):
    """NodeBased's ShuffleCopy routing as Nuke 17's ShuffleCopy. Nuke's input 0 is the node's '2' input (the red2, green2... entries) and
    input 1 its '1' input (red, green...); the pack wires NodeBased's in1 to input 0, so NodeBased's 'in1.r' is Nuke's 'red2' and
    'in2.r' is Nuke's 'red'. Read in Nuke 17.0v6 on 11/20: the default is red2 green2 blue2 alpha, colour from input 0, alpha from 1."""
    p = case.nb_params or {}
    names = {"r": "red", "g": "green", "b": "blue", "a": "alpha"}
    lines = ["node['in'].setValue('rgba')", "node['in2'].setValue('rgba')"]
    for c in "rgba":
        route = str(p.get(f"out1_{c}", f"in1.{c}"))
        if route in ("0", "1"):
            value = "black" if route == "0" else "white"
        elif route[:4] in ("in1.", "in2.") and route[4:] in names:
            value = names[route[4:]] + ("2" if route.startswith("in1.") else "")
        else:
            raise ValueError(f"NodeBased's ShuffleCopy route {route!r} has no ShuffleCopy entry in the pack")
        lines.append(f"node[{names[c]!r}].setValue({value!r})")
    return lines


# Nodes whose settings the pack writes from NodeBased's parameters, because the case lists no Nuke knobs for them.
ROUTING_WRITERS = {"Shuffle": shuffle2_lines, "ShuffleCopy": shufflecopy_lines}


def unwritten_settings(case):
    """NodeBased parameters a case sets away from the node's defaults while giving Nuke no knob at all: the render would show Nuke's
    defaults. Underscored keys are the pack's own (fixtures, stamps) and legacy_* flags pick NodeBased's old behaviour, not a setting
    Nuke has. Empty when the case lists knobs or the pack writes the node's settings itself."""
    if case.nuke_knobs or case.nuke_class in ROUTING_WRITERS:
        return []
    from nodebased import core
    defaults = core.SPECS.get(case.nb_kind, {}).get("params", {})
    return [k for k, v in (case.nb_params or {}).items()
            if not k.startswith("_") and not k.startswith("legacy_") and k in defaults and defaults[k] != v]


def _stamp(case):
    return (case.nb_params or {}).get("_stamp")


def unrenderable(case, table):
    """Why the pack cannot render `case` in Nuke 17 Non-commercial, or "" when it can. Every reason names what is missing."""
    cls = nuke17_class(case)
    if case.nuke_class in NOT_IN_NC:
        return f"Nuke Non-commercial does not offer {case.nuke_class}"
    if case.nuke_class in NEEDS_SEQUENCE:
        return NEEDS_SEQUENCE[case.nuke_class]
    params = case.nb_params or {}
    for key, what in (("_shapes", "Roto shapes"), ("_items", "RotoPaint strokes"), ("_tracks", "Tracker tracks")):
        if key in params:
            return f"the {what} live in the node's curve data, which the pack does not write into Nuke yet"
    stamp = _stamp(case)
    if stamp is not None:
        if fractional_frame(case) is not None and (sequence_plate(case) or stamp.get("curves")):
            return f"the case shows fractional frame {stamp['frame']} on a moving plate or a curve, which the pack's TimeWarp does not combine yet"
        if any(step[0] != "FrameRange" for clip in stamp.get("clips", ()) for step in clip):
            return "an AppendClip input upstream of the node carries a Transform/Shuffle chain, which the pack does not build in Nuke yet"
        if "inputs" in stamp:       # Nuke 17's Python has no input labels (no Node.inputLabel, 11/21), so the FgVecs index needs a probe in the GUI
            return "the node's extra inputs (Kronos's vectors on FgVecs) are not wired in Nuke yet: the input's index is not known from Python"
        if "spots" in stamp:
            return "DustBust's spots live in the node's paint data, which the pack does not write into Nuke yet"
        for param in stamp.get("curves", {}):
            if (case.nuke_class, param) not in CURVE_KNOB:
                return f"the curve on NodeBased's {param} has no Nuke knob in the pack's map"
        if case.nuke_inputs not in (("stamp",), ()):
            return "the stamped case also reads fixture pictures, which the pack does not combine yet"
    elif any(name not in FIXTURE_NAMES for name in case.nuke_inputs):
        return "the case reads a picture the pack does not write: " + ", ".join(n for n in case.nuke_inputs if n not in FIXTURE_NAMES)
    unwritten = unwritten_settings(case)
    if unwritten:
        return (f"the case sets NodeBased's {', '.join(unwritten[:4])} and lists no Nuke knob for them (curves, grids or files the pack "
                "does not write into Nuke yet), so a render would show Nuke's defaults")
    try:
        for knob, value in tuple(case.nuke_knobs) + tuple(implied_knobs(case, table)):
            case_knob_statement(case, knob, value, table)
        if case.nuke_class in ROUTING_WRITERS:
            ROUTING_WRITERS[case.nuke_class](case)
    except ValueError as exc:
        return str(exc)
    return ""


def _stamp_lines(case, stamp):
    """The frame-stamped plate in Nuke: an Expression node at the fixture's format whose red is the frame (held outside the stamp's keys,
    like NodeBased's keyed Constant), green 0.25 or the frame times `green_scale`, blue 0.5 (AppendClip's clips 0.2, 0.3, ...), alpha 1.
    A FrameRange chain follows when the case has one. Returns (lines, the names of the plates by the node's input index)."""
    first_key, last_key = stamp.get("keys", (-90, 100))
    held = f"clamp(frame, {first_key}, {last_key})"
    green = f"{held} * {stamp['green_scale']!r}" if "green_scale" in stamp else "0.25"
    lines, names = [], []
    clips = stamp.get("clips") or [stamp.get("chain", ())]
    for index, steps in enumerate(clips):
        blue = round(0.2 + 0.1 * index, 6) if "clips" in stamp else STAMP_BLUE
        name = f"plate{index}"
        lines.append(f"{name} = nuke.nodes.Expression(expr0={held!r}, expr1={green!r}, expr2={str(blue)!r}, expr3='1')")
        last = name
        for step_index, (kind, params, *_) in enumerate(steps):
            step = f"{name}_{step_index}"
            lines.append(f"{step} = nuke.nodes.FrameRange(first_frame={int(params['first_frame'])}, last_frame={int(params['last_frame'])})")
            lines.append(f"{step}.setInput(0, {last})")
            last = step
        names.append(last)
    return lines, names


def py_script(case, table=None):
    """One case as a Python file for `nuke -t`: Read nodes for the fixtures (raw, so no colour transform) or, for a Time case, the
    frame-stamped plate built in Nuke; the node under test with its knobs (written for each knob's kind) and inputs; a Crop to the node's
    format, so the picture stays inside Non-commercial's 1920x1080 and matches what NodeBased's viewer shows; and a 32-bit float EXR
    Write of all four channels, executed at the case's frame."""
    table = _knob_table() if table is None else table
    reason = unrenderable(case, table)
    if reason:
        raise ValueError(f"{case.id}: {reason}")
    cls = nuke17_class(case)
    stamp = _stamp(case)
    fractional = fractional_frame(case)
    frame = 1 if fractional is not None else (int(stamp.get("frame", 1)) if stamp else 1)
    first, last = (stamp.get("time") or (min(1, frame), max(1, frame))) if stamp else (1, 1)
    first, last = min(int(first), frame), max(int(last), frame)
    width, height = (stamp or {}).get("size", (W, H))
    lines = ["# One case of the NodeBased fidelity pack, written by tools/make_nuke_reference_pack.py. Run: nuke -t THIS_FILE",
             f"# {case.id}: {case.nuke} ({cls} in Nuke 17)  ({case.source})",
             "import os", "import sys", "import nuke", "",
             "try:",
             "    here = os.path.dirname(os.path.abspath(__file__))",
             "except NameError:                      # nuke -t may run the file without defining __file__",
             "    here = os.path.dirname(os.path.abspath(sys.argv[0]))",
             "fixtures = os.path.join(here, os.pardir, 'fixtures')",
             "out_dir = os.environ['NB_RENDER_DIR']",
             "",
             "# Non-commercial Nuke reaches at most 10 nodes per script from Python, so every node is kept as the handle",
             "# nuke.nodes.X() returns; nothing is looked up with nuke.toNode or nuke.allNodes.",
             f"nuke.addFormat('{width} {height} 1 nb_fixture')",
             "nuke.root()['format'].setValue('nb_fixture')",
             f"nuke.root()['first_frame'].setValue({first})", f"nuke.root()['last_frame'].setValue({last})"]
    config = ocio_config(case)
    if config:
        lines += ["nuke.root()['colorManagement'].setValue('OCIO')", "nuke.root()['OCIO_config'].setValue('custom')",
                  f"nuke.root()['customOCIOConfigPath'].setValue(os.path.join(fixtures, {config!r}).replace(os.sep, '/'))"]
    gui_range = None
    if stamp is not None and sequence_plate(case):
        p0, p1 = plate_range(case)
        lines.append(f"plate0 = nuke.nodes.Read(file=os.path.join(fixtures, {case.id + '_plate.####.exr'!r}).replace(os.sep, '/'), "
                     f"first={p0}, last={p1}, origfirst={p0}, origlast={p1}, raw=True)")
        plates = ["plate0"]
        gui_range = (first, last)        # the node's input range is the project's, as NodeBased's is, not the margin the sequence carries
    elif stamp is not None:
        plate_lines, plates = _stamp_lines(case, stamp)
        lines += plate_lines
    else:
        names = tuple(case.nb_inputs) if vectors_on_the_source(case) else tuple(case.nuke_inputs)
        for name in dict.fromkeys(names):
            lines.append(f"read_{name} = nuke.nodes.Read(file=os.path.join(fixtures, {name + '.exr'!r}).replace(os.sep, '/'), "
                         "first=1, last=1, origfirst=1, origlast=1, raw=True)")
        plates = [f"read_{name}" for name in case.nuke_inputs]
        if case.nuke_class in NUKE17_INPUT_ORDER:
            plates = [plates[i] for i in NUKE17_INPUT_ORDER[case.nuke_class]]
        if vectors_on_the_source(case):
            v = VECTOR_LAYER
            lines.append(f"nuke.Layer({v!r}, [{', '.join(repr(f'{v}.{c}') for c in ('red', 'green', 'blue', 'alpha'))}])")
            lines.append("vectors = nuke.nodes.Copy(" + ", ".join(f"from{i}='rgba.{c}', to{i}='{v}.{c}'"
                                                            for i, c in enumerate(('red', 'green', 'blue', 'alpha'))) + ")")
            lines += [f"vectors.setInput(0, read_{case.nb_inputs[0]})", f"vectors.setInput(1, read_{case.nb_inputs[1]})"]
            plates = ["vectors"]
    lines.append(f"node = nuke.nodes.{cls}()")
    lines += [f"node.setInput({index}, {plate})" for index, plate in enumerate(plates)]
    if stamp is not None and sequence_plate(case):
        for slot in (stamp.get("inputs") or {}):
            lines.append(f"read_{slot} = nuke.nodes.Read(file=os.path.join(fixtures, {case.id + '_' + slot + '.exr'!r}).replace(os.sep, '/'), "
                         "first=1, last=1, origfirst=1, origlast=1, raw=True)")
            lines.append(f"node.setInput(next(i for i in range(node.maxInputs()) if "
                         f"node.inputLabel(i).lower().replace(' ', '') == {slot.lower()!r}), read_{slot})")
    for knob in GUI_INPUT_RANGE.get(cls, ()) if plates else ():
        end = "firstFrame" if knob.endswith("first") else "lastFrame"
        if gui_range is not None:
            lines.append(f"node[{knob!r}].setValue({gui_range[0] if end == 'firstFrame' else gui_range[1]})          # the project range")
        else:
            lines.append(f"node[{knob!r}].setValue(node.input(0).{end}())          # as the GUI sets it on creation")
    statements = [case_knob_statement(case, knob, value, table) for knob, value in tuple(case.nuke_knobs) + tuple(implied_knobs(case, table))]
    if vectors_on_the_source(case):
        statements = [s.replace("'rgba.", f"'{VECTOR_LAYER}.").replace(" rgba.", f" {VECTOR_LAYER}.") if s.startswith("node['uv']") else s
                      for s in statements]
        if not any(s.startswith("node['uv']") for s in statements):
            statements.append(f"node['uv'].setValue('{VECTOR_LAYER}.red {VECTOR_LAYER}.green')")
    lines += statements
    if case.nuke_class in ROUTING_WRITERS:
        lines += ROUTING_WRITERS[case.nuke_class](case)
    for param, (interpolation, keys) in (stamp or {}).get("curves", {}).items():
        knob = CURVE_KNOB[(case.nuke_class, param)]
        lines.append(f"node[{knob!r}].setAnimated()")
        lines += [f"node[{knob!r}].setValueAt({float(v)!r}, {float(f)!r})" for f, v in keys]
        lines.append(f"node[{knob!r}].animation(0).changeInterpolation(node[{knob!r}].animation(0).keys(), "
                     f"{CURVE_INTERPOLATION.get(interpolation, 'nuke.LINEAR')})")
    shown = "node"
    if fractional is not None:
        lines += ["warp = nuke.nodes.TimeWarp()          # asks the node for the case's fractional frame; Nuke renders whole frames",
                  "warp.setInput(0, node)", f"warp['lookup'].setValue({fractional!r})", "warp['filter'].setValue('none')"]
        shown = "warp"
    lines += ["crop = nuke.nodes.Crop()",
              "crop['box'].setValue([0, 0, 0, 0])",
              "crop['box'].setExpression('input.format.x', 0)", "crop['box'].setExpression('input.format.y', 1)",
              "crop['box'].setExpression('input.format.r', 2)", "crop['box'].setExpression('input.format.t', 3)",
              f"crop.setInput(0, {shown})",
              f"write = nuke.nodes.Write(file=os.path.join(out_dir, {case.id + '.exr'!r}).replace(os.sep, '/'), "
              "file_type='exr', datatype='32 bit float', compression='none', channels='rgba', raw=True)",
              "write.setInput(0, crop)",
              f"nuke.execute(write, {frame}, {frame})"]
    return "\n".join(lines) + "\n"


RENDER_ALL = """#!/bin/sh
# Render every script of the pack with Nuke's terminal mode, one frame each, into the folder given as the first argument.
# NUKE selects the executable (default: nuke); for Nuke Non-commercial point it at the wrapper that adds --nc.
# `nuke -t` runs a Python file, which Non-commercial allows; `nuke -x` would need an encrypted .nknc script instead.
set -u
out="${1:?usage: render_all.sh RENDER_DIR}"
nuke="${NUKE:-nuke}"
here="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$out"
export NB_RENDER_DIR="$(cd "$out" && pwd)"
failed=0
for script in "$here"/scripts/*.py; do
    id="$(basename "$script" .py)"
    rm -f "$out/$id.exr"
    if "$nuke" -t "$script" </dev/null >"$out/$id.log" 2>&1 && [ -s "$out/$id.exr" ]; then
        echo "rendered $id"
    else
        echo "FAILED   $id (see $out/$id.log)" >&2
        failed=$((failed + 1))
    fi
done
echo "$failed script(s) failed"
[ "$failed" -eq 0 ]
"""

README = """Nuke reference pack for tools/audit_nuke_fidelity.py

What it is: {n} cases. Each is a small Python script (fixture images -> one Nuke node -> a 32-bit float EXR), built from the
same fixtures and the same knob values the audit's formula cases use. Rendering them with Nuke gives the 'pixel truth' column.

To render (Nuke 17, terminal mode):   NUKE=/path/to/nuke ./render_all.sh RENDER_DIR
  Nuke Non-commercial: NUKE is the wrapper that runs `Nuke17.0 --nc`. The first run on a machine asks for the one-time
  Foundry sign-in; until that is done every case fails and its log says "No tokens found".
To score:   python tools/audit_nuke_fidelity.py --reference-dir RENDER_DIR

Caveats
- Non-commercial limits that shaped the scripts: only `nuke -t` Python files run (command-line renders accept encrypted .nknc
  scripts only), at most 10 nodes per script can be reached from Python (each script keeps the handles nuke.nodes.X returns),
  and output is capped at 1920x1080.
- The scripts use Nuke's documented Python API; the first render on a machine is the first time they run. If a script fails, the
  log next to its render says why; fix the generator (tools/make_nuke_reference_pack.py), not the render.
- Read and Write are set to raw, so no colour transform touches the numbers. Check the first EXR of a Merge case against
  fixtures/A.exr and fixtures/B.exr by eye before trusting the rest.
- Input order follows Nuke's: input 0 is B on Merge, Keymix; input 1 is A; the mask is the last input.
- Nothing rendered belongs in the repository.
"""


def write_pack(pack, table=None):
    """Write the pack. Cases that are not pixel behaviours or nodes NodeBased lacks are left out (MANIFEST "skipped"); cases Nuke 17
    Non-commercial cannot render get no script and are listed with the reason (MANIFEST "not_renderable")."""
    pack = Path(pack)
    table = _knob_table() if table is None else table
    (pack / "fixtures").mkdir(parents=True, exist_ok=True)
    (pack / "scripts").mkdir(parents=True, exist_ok=True)
    from nodebased.media import write_exr
    fixtures = make_fixtures()
    digests = {}
    for name, arr in fixtures.items():
        write_exr(pack / "fixtures" / f"{name}.exr", arr, bits="float", compression="none")
        digests[name] = hashlib.sha256(np.ascontiguousarray(arr, np.float32).tobytes()).hexdigest()
    manifest = {"size": [W, H], "fixtures": digests, "tolerance": TOLERANCE, "cases": [], "skipped": [], "not_renderable": []}
    for case in CASES:
        if case.id.endswith(SKIPPED_SUFFIXES):
            manifest["skipped"].append({"id": case.id, "nuke": case.nuke, "reason": case.statement or case.id})
            continue
        reason = unrenderable(case, table)
        if reason:
            manifest["not_renderable"].append({"id": case.id, "nuke": case.nuke, "nuke_class": nuke17_class(case), "reason": reason})
            continue
        (pack / "scripts" / f"{case.id}.py").write_text(py_script(case, table), encoding="utf-8")
        for name, path in case_files(case).items():           # a LUT the case made for itself, beside the fixtures
            shutil.copyfile(path, pack / "fixtures" / name)
        if sequence_plate(case):                              # the moving plate and extra inputs, as NodeBased makes them
            for name, arr in plate_sequences(case).items():
                frame_name = name
                if "_plate." in name:
                    stem, number, ext = name.rsplit(".", 2)
                    frame_name = f"{stem}.{int(number):04d}.{ext}" if int(number) >= 0 else f"{stem}.-{abs(int(number)):03d}.{ext}"
                write_exr(pack / "fixtures" / frame_name, arr, bits="float", compression="none")
        manifest["cases"].append({"id": case.id, "nuke": case.nuke, "nuke_class": nuke17_class(case),
                                  "nuke_knobs": [[k, v] for k, v in case.nuke_knobs], "nuke_inputs": list(case.nuke_inputs),
                                  "nodebased_kind": case.nb_kind, "nodebased_params": case.nb_params, "source": case.source})
    (pack / "MANIFEST.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n", encoding="utf-8")
    script = pack / "render_all.sh"
    script.write_text(RENDER_ALL, encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    (pack / "README.txt").write_text(README.format(n=len(manifest["cases"])), encoding="utf-8")
    return len(manifest["cases"])


# ------------------------------------------------------------------------------------------------------------ the loader
def load_render(folder, case_id):
    """A rendered EXR as a float32 (H, W, 4) array in file order (top row first), or None when the file is not there."""
    path = Path(folder) / f"{case_id}.exr"
    if not path.exists():
        return None
    import OpenImageIO as oiio
    source = oiio.ImageInput.open(str(path))
    if source is None:
        raise ValueError(f"cannot read {path}: {oiio.geterror()}")
    try:
        spec = source.spec()
        pixels = source.read_image("float")
    finally:
        source.close()
    pixels = np.asarray(pixels, np.float32).reshape(spec.height, spec.width, spec.nchannels)
    if spec.nchannels < 4:          # a render without alpha (the first NC pass wrote rgb only): alpha is unknown, so it reads as NaN
        pixels = np.concatenate([pixels, np.full(pixels.shape[:2] + (4 - spec.nchannels,), np.nan, np.float32)], axis=2)
    return place_in_display_window(pixels[..., :4], (spec.x, spec.y), (spec.full_x, spec.full_y, spec.full_width, spec.full_height))


def place_in_display_window(pixels, data_origin, display):
    """An EXR's data window placed in its display window (the format), zero where the data does not reach: the picture NodeBased's viewer
    shows, which is what the cases compare. `data_origin` is (x, y) of the data window, `display` (x, y, width, height); top row first."""
    dx, dy, dw, dh = display
    if (data_origin[0], data_origin[1], pixels.shape[1], pixels.shape[0]) == (dx, dy, dw, dh):
        return pixels
    canvas = np.zeros((dh, dw, pixels.shape[2]), np.float32)
    x0, y0 = data_origin[0] - dx, data_origin[1] - dy
    sx, sy = max(0, -x0), max(0, -y0)
    tx, ty = max(0, x0), max(0, y0)
    w = min(pixels.shape[1] - sx, dw - tx)
    h = min(pixels.shape[0] - sy, dh - ty)
    if w > 0 and h > 0:
        canvas[ty:ty + h, tx:tx + w] = pixels[sy:sy + h, sx:sx + w]
    return canvas


def compare_arrays(ours, theirs):
    """Per channel maximum absolute error and the worst pixel, plus pass/fail against TOLERANCE."""
    if ours.shape != theirs.shape:
        return {"status": "fail", "note": f"shape {ours.shape} vs {theirs.shape}", "channels": {}}
    err = np.abs(ours.astype(np.float64) - theirs.astype(np.float64))
    channels = {}
    ok = True
    for c, name in enumerate(CHANNELS):
        e = err[..., c]
        y, x = np.unravel_index(int(np.argmax(e)), e.shape)
        channels[name] = {"max_abs_error": float(e.max()), "worst_pixel": [int(y), int(x)]}
        ok &= bool(np.all(e <= TOLERANCE["atol"] + TOLERANCE["rtol"] * np.abs(theirs[..., c])))
    return {"status": "pass" if ok else "fail", "note": "", "channels": channels}


def compare_case(case, folder):
    """Compare NodeBased's output for `case` with the Nuke render in `folder`; None when there is no render for the case."""
    theirs = load_render(folder, case.id)
    if theirs is None:
        return None
    try:
        ours = run_nodebased(case)
    except Exception as exc:  # noqa: BLE001 - a case NodeBased cannot run is a finding against the render, not a crash
        return {"id": case.id, "status": "error", "note": f"NodeBased raised {type(exc).__name__}: {str(exc)[:80]}", "channels": {}}
    if ours is None:
        return {"id": case.id, "status": "fail", "note": "NodeBased cannot express this setting: " + case.inexpressible, "channels": {}}
    out = compare_arrays(ours, theirs)
    out["id"] = case.id
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("pack_dir", help="folder to write the pack into (created)")
    args = ap.parse_args(argv)
    print(f"wrote {write_pack(args.pack_dir)} cases into {args.pack_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
