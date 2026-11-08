#!/usr/bin/env python3
"""How closely do NodeBased's 2D nodes match the real Nuke nodes? One 0 to 100 number per node, and an overall number.

Run from a checkout:

    QT_QPA_PLATFORM=offscreen python tools/audit_nuke_fidelity.py            # print the report
    QT_QPA_PLATFORM=offscreen python tools/audit_nuke_fidelity.py --write    # rewrite docs/PARITY_2D_FIDELITY.md
    QT_QPA_PLATFORM=offscreen python tools/audit_nuke_fidelity.py --check    # fail if the committed report drifted
    ... --reference-dir DIR      # also score pixel truth against a folder of Nuke renders (see make_nuke_reference_pack.py)

Inputs, all committed:

* docs/parity/nuke_nodes.json         what the Nuke 17.0 Reference Guide lists for every node row of docs/PARITY_2D.md
* docs/parity/nodebased_to_nuke.json  every NodeBased knob and socket paired with its Nuke counterpart, or marked extra
* docs/PARITY_2D.md                   the rows: group, daily-use rank, status
* tools/nuke_fidelity_cases.py        the fixture images and the formula cases

The scorer matches nothing by name on its own: a NodeBased knob counts for a Nuke knob only when the mapping file pairs them.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

NUKE_JSON = ROOT / "docs" / "parity" / "nuke_nodes.json"
MAP_JSON = ROOT / "docs" / "parity" / "nodebased_to_nuke.json"
DOC = ROOT / "docs" / "PARITY_2D.md"
REPORT = ROOT / "docs" / "PARITY_2D_FIDELITY.md"

GROUPS = ("Image", "Draw", "Time", "Channel", "Color", "Filter", "Keyer", "Merge", "Transform", "Metadata", "Other")
# Component weights in a node's score. A component a node does not have (no Nuke inputs, no formula cases, no renders yet)
# is left out and the rest are renormalised.
# The requirements of November 2026 (docs/PARITY_2D_FIDELITY.md, "How a node is scored"): knobs that act, Nuke's layout,
# per-channel splits, and behaviour that counts against a node until it is checked.
WEIGHTS = {"knobs": 30, "acts": 20, "defaults": 10, "sockets": 10, "chrome": 10, "math": 25, "layout": 10, "splits": 10, "pixels": 30}
# The formula the report used from 2026-10-25 to 2026-11-02, still printed beside the new one so the history stays readable.
PREVIOUS_WEIGHTS = {"knobs": 40, "defaults": 10, "sockets": 10, "chrome": 15, "math": 25, "pixels": 30}
KNOB_WEIGHT = {True: 2, False: 1}                   # primary (main tab) knobs weigh 2, secondary knobs 1
CHROME_ROLES = ("channels", "mask_channel", "mask_invert", "mask_inject", "mask_fringe", "unpremult", "unpremult_invert", "mix", "bbox")
CHROME_LABEL = {"channels": "channels selector", "mask_channel": "mask channel", "mask_invert": "mask invert", "mask_inject": "mask inject",
                "mask_fringe": "mask fringe", "unpremult": "(un)premult by", "unpremult_invert": "(un)premult invert", "mix": "mix", "bbox": "bbox handling"}
ALLCLOSE = dict(rtol=1e-5, atol=1e-5)
GAP_LIST = 30


# ------------------------------------------------------------------------------------------------------------- inputs
def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def names_of(cell):
    cell = re.sub(r"\s*\(no Nuke node\)", "", cell)
    if re.match(r"^LensDistortion \((.+)\)$", cell):
        return ["LensDistortion"]
    return [n.strip() for n in cell.split(" / ")]


def doc_rows(path=DOC):
    """[(group, rank, cell, status)] for every table row of docs/PARITY_2D.md."""
    rows, heading = [], ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        m = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if m and m.group(3) in ("supported", "partial", "missing", "n/a") and heading in GROUPS:
            rows.append((heading, int(m.group(1)), m.group(2), m.group(3)))
    return rows


def pair_key(group, name):
    return "Multiply (Merge)" if (group, name) == ("Merge", "Multiply") else name


# ----------------------------------------------------------------------------------------------------------- defaults
def document_default(path):
    """The value the document field `path` ("time.first", "settings.color.view", "settings.viewer.proxy") has in a new
    project. A path that names no field raises KeyError, so a mapping with a typo fails instead of scoring."""
    from nodebased import core
    doc = core.demo_document()
    if path == "settings.viewer.proxy":                     # optional key: absent means the full-size tier
        return core.viewer_proxy(doc)
    if path.startswith("settings.project."):                # optional group: a field not set holds its default
        from nodebased import projectsettings
        return projectsettings.DEFAULTS[path.split(".", 2)[2]]
    value = doc
    for part in path.split("."):
        value = value[part]
    return value


def _norm(text):
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


_ALIAS = {"all": "rgba"}


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def compare_default(nb_value, nuke_value, value_map=None):
    """True/False when the two defaults can be compared, None when they cannot (different kinds of value)."""
    if value_map and isinstance(nb_value, str) and nb_value in value_map:
        nb_value = value_map[nb_value]
    if isinstance(nuke_value, list):
        if isinstance(nb_value, list) and len(nb_value) == len(nuke_value) and all(_is_num(a) and _is_num(b) for a, b in zip(nb_value, nuke_value)):
            return all(abs(a - b) <= 1e-6 * max(1.0, abs(b)) for a, b in zip(nb_value, nuke_value))
        return None
    if isinstance(nb_value, list):
        if _is_num(nuke_value) and all(_is_num(a) for a in nb_value):
            return all(abs(a - nuke_value) <= 1e-6 * max(1.0, abs(nuke_value)) for a in nb_value)
        return None
    if isinstance(nuke_value, bool):
        if isinstance(nb_value, (bool, int, float)):
            return bool(nb_value) == nuke_value
        return None
    if _is_num(nuke_value):
        if isinstance(nb_value, bool):
            return bool(nb_value) == bool(nuke_value)
        if _is_num(nb_value):
            return abs(nb_value - nuke_value) <= 1e-6 * max(1.0, abs(nuke_value))
        return None
    if isinstance(nuke_value, str):
        if isinstance(nb_value, str):
            a, b = _norm(nb_value), _norm(nuke_value)
            return _ALIAS.get(a, a) == _ALIAS.get(b, b)
        if isinstance(nb_value, bool) and nuke_value.lower() in ("on", "off", "true", "false", "enabled", "disabled"):
            return nb_value == (nuke_value.lower() in ("on", "true", "enabled"))
        return None
    return None


# ------------------------------------------------------------------------------------------------------------ formulas
_FIXTURES = None
_CASE_CACHE = {}


def run_case(case):
    """Run one formula case on NodeBased's evaluator kernel. Returns a dict with status pass/fail/inexpressible/error."""
    from nuke_fidelity_cases import make_fixtures, run_nodebased
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = make_fixtures()
    if case.id in _CASE_CACHE:
        return _CASE_CACHE[case.id]
    res = {"id": case.id, "nuke": case.nuke, "source": case.source, "statement": case.statement, "unscored": case.unscored,
           "status": "inexpressible", "max_abs_error": None, "worst": None, "note": case.inexpressible}
    if case.unscored:
        res["status"] = "not scored"
    elif case.nb_params is not None:
        try:
            out = run_nodebased(case, _FIXTURES)
            ref, valid = case.expected(_FIXTURES)
            err = np.where(valid, np.abs(out - ref), 0.0)
            tol = max(ALLCLOSE["atol"], float(getattr(case, "tolerance", 0.0))) + ALLCLOSE["rtol"] * np.abs(ref)
            ok = bool(np.all(np.where(valid, err <= tol, True)))
            # Rounded before the worst pixel is picked, so float noise between machines cannot reorder near-equal errors.
            rounded = np.round(err, 4)
            res["max_abs_error"] = float(rounded.max())
            res["worst"] = [int(i) for i in np.unravel_index(int(np.argmax(rounded)), rounded.shape)]
            res["status"] = "pass" if ok else "fail"
            res["note"] = ""
        except Exception as exc:  # noqa: BLE001 - a case that cannot run is a finding, not a crash
            res["status"] = "error"
            res["note"] = f"{type(exc).__name__}: {str(exc)[:80]}"
    _CASE_CACHE[case.id] = res
    return res


def pixel_truth(cases, reference_dir):
    """Per-case comparison of NodeBased against Nuke renders in `reference_dir`; None when no render exists for the cases."""
    from make_nuke_reference_pack import compare_case
    out = []
    for case in cases:
        got = compare_case(case, reference_dir)
        if got is not None:
            out.append(got)
    return out or None


# -------------------------------------------------------------------------------------------------------------- scoring
def score_pair(key, pair, nuke, reference_dir=None):
    """The component fractions and the 0..100 fidelity of one Nuke node against its NodeBased counterpart."""
    from nodebased import core
    from nuke_fidelity_cases import cases_for, make_fixtures
    import nuke_fidelity_requirements as req
    global _FIXTURES
    if _FIXTURES is None:
        _FIXTURES = make_fixtures()
    res = {"nuke": key, "nodebased": pair.get("nodebased") or pair.get("document_settings"), "components": {}, "detail": {}}
    knobs = [k for k in nuke["knobs"] if k["counted"] and not k["role"].startswith("chrome:")]
    chrome = {}
    for k in nuke["knobs"]:
        if k["role"].startswith("chrome:"):
            chrome.setdefault(k["role"][7:], []).append(k["name"])
    sockets = list(dict.fromkeys(i["name"] for i in nuke["inputs"]))
    all_cases = cases_for(key)
    cases = [c for c in all_cases if not c.unscored]
    document = pair.get("document_settings") is not None
    if pair.get("nodebased") is None and not document:
        mapped, entries = set(), []
        spec = None
    elif document:
        # Root: the Nuke node is the project's own settings, which live in document fields rather than a node's params.
        spec = None
        entries = [e for e in pair["knobs"] if e["kind"] in ("same", "renamed", "components", "panel")]
        mapped = {n for e in entries for n in e["nuke"]}
    else:
        spec = core.SPECS[pair["nodebased"]]
        entries = [e for e in pair["knobs"] if e["kind"] in ("same", "renamed", "components", "panel")]
        mapped = {n for e in entries for n in e["nuke"]}
    # knob coverage
    if knobs:
        total = sum(KNOB_WEIGHT[k["primary"]] for k in knobs)
        have = sum(KNOB_WEIGHT[k["primary"]] for k in knobs if k["name"] in mapped)
        res["components"]["knobs"] = have / total
        res["detail"]["missing_primary"] = [k["name"] for k in knobs if k["primary"] and k["name"] not in mapped]
        res["detail"]["missing_secondary"] = [k["name"] for k in knobs if not k["primary"] and k["name"] not in mapped]
        res["detail"]["knob_counts"] = (sum(1 for k in knobs if k["name"] in mapped), len(knobs))
    # defaults
    if pair.get("nodebased") is not None or document:
        skip = set(pair.get("skip_default", []))
        vmap = pair.get("value_map", {})
        first = {}
        for e in entries:
            if e["kind"] == "panel":
                continue
            for n in e["nuke"]:
                first.setdefault(n, e)
        comparable = matches = 0
        mismatches = []
        for k in nuke["knobs"]:
            if not (k["counted"] or k["role"].startswith("chrome:")) or k["name"] not in first or k["name"] in skip or k["default"] is None:
                continue
            e = first[k["name"]]
            if len(e["nuke"]) != 1:
                continue
            vals = [document_default(p) if document else spec["params"][p] for p in e["nodebased"]]
            nb_value = vals if len(vals) > 1 else vals[0]
            ok = compare_default(nb_value, k["default"], vmap.get(k["name"]))
            if ok is None:
                continue
            comparable += 1
            matches += bool(ok)
            if not ok:
                mismatches.append((k["name"], nb_value, k["default"]))
        if comparable:
            res["components"]["defaults"] = matches / comparable
            res["detail"]["default_mismatches"] = mismatches
            res["detail"]["default_counts"] = (matches, comparable)
    # sockets
    if sockets:
        present = {i["nuke"] for i in pair.get("inputs", []) if i["kind"] == "mapped"}
        res["components"]["sockets"] = sum(1 for s in sockets if s in present) / len(sockets)
        res["detail"]["missing_sockets"] = [s for s in sockets if s not in present]
    # chrome
    roles = [r for r in CHROME_ROLES if r in chrome]
    if roles:
        have = [r for r in roles if any(n in mapped for n in chrome[r])]
        res["components"]["chrome"] = len(have) / len(roles)
        res["detail"]["missing_chrome"] = [r for r in roles if r not in have]
    # knobs that act: each paired knob, moved off its default, changes the kernel's pixels on the fixture
    kind = pair.get("nodebased")
    tap = kind in req.TAPS
    runs = kind is not None and not document and not tap and req.probe_acts(kind, [], _FIXTURES) is not None
    if tap:
        res["detail"]["tap"] = True
    if runs and knobs:
        frac, detail = req.acts_component(key, kind, entries, knobs, _FIXTURES, KNOB_WEIGHT)
        res["detail"].update(detail)
        if frac is not None:
            res["components"]["acts"] = frac
    # Nuke's layout: knob order and main-tab labels
    res["panel_rows"] = res["panel_types"] = None
    if kind is not None and not document and knobs:
        frac, detail = req.layout_component(kind, entries, knobs)
        res["detail"].update(detail)
        # Since 2026-11-04 the layout fraction is four equal quarters: knob order, main-tab labels, shared rows and widget types;
        # a quarter with nothing to count is left out and the rest renormalised. `layout_old` keeps the two-half form for the report.
        rows, types = panel_shape_metrics(nuke, pair, kind, key)
        res["panel_rows"], res["panel_types"] = rows, types
        quarters = []
        if detail.get("layout_order"):
            quarters.append(detail["layout_order"][0] / detail["layout_order"][1])
        if detail.get("layout_labels") and detail["layout_labels"][1]:
            quarters.append(detail["layout_labels"][0] / detail["layout_labels"][1])
        quarters += [m["score"] for m in (rows, types) if m["score"] is not None]
        res["detail"]["layout_old"] = frac
        if quarters:
            res["components"]["layout"] = sum(quarters) / len(quarters)
    # per-channel splits (Nuke's AColor and WH knobs, docs/parity/nuke_split_knobs.json)
    if not document:
        frac, detail = req.splits_component(key, kind, entries, _split_table())
        res["detail"].update(detail)
        if frac is not None:
            res["components"]["splits"] = frac
    # documented math. A node whose kernel runs on the fixture and that has no scored case has unchecked behaviour: it scores 0
    # here (the previous formula left it out). A node whose every case is listed unscored, with a reason, stays out.
    if cases:
        results = [run_case(c) for c in cases]
        passed = sum(1 for r in results if r["status"] == "pass")
        res["components"]["math"] = passed / len(results)
        res["detail"]["math"] = (passed, len(results))
        res["detail"]["math_failed"] = [r["id"] for r in results if r["status"] != "pass"]
    elif runs and not all_cases and (knobs or sockets or chrome):
        res["components"]["math"] = 0.0
        res["detail"]["math_unchecked"] = True
    # pixel truth against Nuke renders
    truth = pixel_truth([c for c in cases_for(key)], reference_dir) if reference_dir else None
    if truth:
        res["components"]["pixels"] = sum(1 for t in truth if t["status"] == "pass") / len(truth)
        res["detail"]["pixels"] = truth
    live = {c: f for c, f in res["components"].items()}
    if pair.get("nodebased") is None and not document:
        res["score"] = 0.0 if (knobs or sockets or chrome) else None
    elif live:
        res["score"] = 100.0 * sum(WEIGHTS[c] * f for c, f in live.items()) / sum(WEIGHTS[c] for c in live)
    else:
        res["score"] = None
    old = {c: f for c, f in live.items() if c in PREVIOUS_WEIGHTS and not (c == "math" and res["detail"].get("math_unchecked"))}
    if res["score"] is None or not old:
        res["score_previous"] = res["score"]
    else:
        res["score_previous"] = 100.0 * sum(PREVIOUS_WEIGHTS[c] * f for c, f in old.items()) / sum(PREVIOUS_WEIGHTS[c] for c in old)
    # the same node with the layout fraction of the two-half form used until 2026-11-04
    alt = dict(live)
    if res["detail"].get("layout_old") is None:
        alt.pop("layout", None)
    else:
        alt["layout"] = res["detail"]["layout_old"]
    if res["score"] is not None and alt and pair.get("nodebased") is not None:
        res["score_layout_old"] = 100.0 * sum(WEIGHTS[c] * f for c, f in alt.items()) / sum(WEIGHTS[c] for c in alt)
    else:
        res["score_layout_old"] = res["score"]
    res["document"] = document
    res["pixel_column"] = ("measured" if truth else ("unmeasured" if cases_for(key) else "no fixture"))
    return res


_SPLITS = None


def _split_table():
    import nuke_fidelity_requirements as req
    global _SPLITS
    if _SPLITS is None:
        _SPLITS = req.load_split_table()
    return _SPLITS


def build_units(nuke_doc, map_doc, reference_dir=None):
    """Score every Nuke node of every non-n/a row. Returns (units, skipped)."""
    nodes, pairs = nuke_doc["nodes"], map_doc["pairs"]
    units, skipped = [], []
    for group, rank, cell, status in doc_rows():
        names = [n for n in names_of(cell) if n != "LightMixer"]
        if status == "n/a":
            for n in names:
                skipped.append((group, rank, n, "n/a in PARITY_2D.md"))
            continue
        members = [n for n in names if pair_key(group, n) in pairs]
        for n in names:
            key = pair_key(group, n)
            if key not in pairs:
                skipped.append((group, rank, n, "no pairing in nodebased_to_nuke.json"))
                continue
            res = score_pair(key, pairs[key], nodes[key], reference_dir)
            if res["panel_rows"] is None:   # a node without a layout still reports its measurements
                res["panel_rows"], res["panel_types"] = panel_shape_metrics(nodes[key], pairs[key], pairs[key].get("nodebased"), key)
            res.update(group=group, rank=rank, name=n, status=status, weight=(1.0 / rank) / len(members))
            units.append(res)
    return units, skipped


_READ_KNOB_TYPES = {
    "file": "file", "localizationPolicy": "enum", "updateLocalization": "button", "format": "enum",
    "proxy": "file", "proxy_format": "enum", "first": "int", "before": "enum", "last": "int",
    "after": "enum", "frame_mode": "enum", "frame": "string", "origfirst": "int",
    "origlast": "int", "on_error": "enum", "reload": "button", "colorspace": "enum",
    "premultiplied": "bool", "raw": "bool", "auto_alpha": "bool",
}
_READ_NODE_WIDGETS = {
    "path": "file", "localizationPolicy": "enum", "read_format": "enum", "proxy": "file",
    "proxy_format": "enum", "first": "int", "before": "enum", "last": "int", "after": "enum",
    "frame_mode": "enum", "frame_expr": "string", "origfirst": "int", "origlast": "int",
    "missing": "enum", "reload": "button", "colorspace": "enum", "premultiplied": "bool",
    "raw": "bool", "auto_alpha": "bool", "key1": "string", "value1": "string", "key2": "string",
    "value2": "string", "key3": "string", "value3": "string", "key4": "string", "value4": "string",
}


_COUNT_WORDS = {"frame", "frames", "first", "last", "count", "samples", "sample", "iterations", "iteration", "index", "seed", "octaves",
                "octave", "divisions", "division", "steps", "step", "levels", "level", "passes", "pass", "number", "num", "layers", "rows",
                "columns", "points", "bits", "cycles", "repeat", "interval", "period"}
# Knobs the guide gives as plain numbers that count frames although no word of their name says so (Tracker's smoothT, smoothR and smoothS
# average the track over that many frames; Switch's which picks an input by index, where Dissolve's is a mix). Panel sweep S2, 2026-11-04.
# Panel sweep S3 (2026-11-04): AppendClip's dissolve is the length of the cross-fade in frames; ContactSheet's width, height and gap are whole
# pixels (Nuke's Resolution is a width/height pair of integers and its gap an Int_Knob), so none of them can be a fractional slider.
_COUNT_KNOBS = {("Tracker", "smoothT"), ("Tracker", "smoothR"), ("Tracker", "smoothS"), ("Tracker", "max_iter"), ("Switch", "which"), ("AppendClip", "dissolve"),
                ("ContactSheet", "width"), ("ContactSheet", "height"), ("ContactSheet", "gap")}
# Panel sweep S4 (2026-11-04): TimeEcho's framesbehind counts frames and ZDefocus's blades counts the aperture's blades (whole numbers in Nuke's
# Int_Knobs). _FLOAT_KNOBS are guide numbers whose name holds a count word although the value is a fraction or a time: IBKGizmo's level is the
# screen range (a multiplier, default 1), Kronos's timingFrame2 is the frame the timing curve asks for, where a fractional frame is interpolated.
# TimeDissolve's in and out are the first and last frame of the dissolve (defaults 1 and 10).
_COUNT_KNOBS |= {("TimeEcho", "framesbehind"), ("ZDefocus", "blades"), ("TimeDissolve", "in"), ("TimeDissolve", "out")}
# Panel sweep S5 (2026-11-07): SmartVector's matteDilation and VectorBlur's grow_bbox are whole pixels (Int_Knobs in Nuke), the guide's defaults 50 and 0 being
# the only hint; Sparkles's ray_nb ("rays Nb", default 50) is a count of rays. All three already show as integer fields.
_COUNT_KNOBS |= {("SmartVector", "matteDilation"), ("VectorBlur", "grow_bbox"), ("Sparkles", "ray_nb")}
# Roto's Motion Blur tab exposes sample counts (whole-number knobs), as do the per-shape settings.
_COUNT_KNOBS |= {(kind, "global_motionblur") for kind in ("Roto", "RotoPaint")}
_COUNT_KNOBS |= {(kind, knob) for kind in ("Roto", "RotoPaint") for knob in ("motionblur", "lifetime_start", "lifetime_end")}
_FLOAT_KNOBS = {("IBKGizmo", "level"), ("Kronos", "timingFrame2"), ("OFlow", "timingFrame2")}
# Knobs whose Nuke widget the guide's default cannot tell: IBKColor's Size (a pixel size, default 10) and multi ("patch black", default 0, which
# the gizmo may show as a check box or a number). Left unmeasured and counted as such, as the rule asks.
_UNSURE_KNOBS = {("IBKColor", "Size"), ("IBKColor", "multi")}
# The Viewer's knobs are viewer state, not node params; this is the control each one has (the Viewer settings window, nodebased/viewersettings.py,
# or the viewer strip: gain and gamma spin boxes, the zebra check box, the proxy tier, mask and safe-zone dropdowns).
_VIEWER_WIDGETS = {
    "frame_range": "string", "frame_range_lock": "bool", "frame_increment": "int", "clip_warning": "bool", "gain": "float",
    "gamma": "float", "rgb_only": "bool", "downrez": "enum", "ignore_pixel_aspect": "bool", "zoom_lock": "bool",
    "show_overscan": "bool", "full_frame_processing": "bool", "overscan": "int", "safe_zone": "enum", "format_center": "bool",
    "masking_ratio": "enum", "masking_mode": "enum", "center_fstop": "float", "shutter_angle": "float", "useGPUForViewer": "bool",
}
_COUNT_SUFFIXES = ("samples", "frames", "frame", "count", "iterations", "index", "seed", "octaves", "steps", "levels", "divisions")


def number_knob_kind(name, default):
    """The widget a guide `number` knob needs: `int` (an integer field, no slider) when the guide default is an integer and the knob
    counts, indexes or names a frame; `float` (a slider with a value field, Nuke's Double_Knob) when the default is a fraction or the
    name is no count; `unknown` (left unmeasured, still counted) when the guide gives no numeric default."""
    if name in ("render_res",):
        return "int"  # the guide's whole-number render resolutions are Int_Knobs (plain integer fields)
    if isinstance(default, bool) or not isinstance(default, (int, float)):
        return "unknown"
    if isinstance(default, float) and not default.is_integer():
        return "float"
    words = set(re.findall(r"[a-z]+", re.sub(r"(?<=[a-z])(?=[A-Z])", " ", name).lower()))
    counts = name.lower() == "corners" or bool(words & _COUNT_WORDS) or name.lower().endswith(_COUNT_SUFFIXES)
    return "int" if counts else "float"


def _widget_kind(group, param):
    """The widget a param gets: its group's kind, or for a row group the kind named for that param."""
    if group.kind == "split" and param != group.params[0]:
        return group.members[group.params.index(param) - 1]     # a check box beside the split knob
    return group.members[group.params.index(param)] if group.kind == "row" else group.kind


def panel_shape_metrics(nuke_node, pair, node_type, key=None):
    """Measure shared guide rows and widget-kind matches; these lines do not affect fidelity."""
    from nodebased.knobs import KnobGroup, knob_layout
    from nodebased.ops2d_chrome import is_open_choice
    import nuke_fidelity_requirements as req

    knobs = nuke_node.get("knobs", [])
    entries = [e for e in pair.get("knobs", []) if e.get("kind") in ("same", "renamed", "components", "panel")]
    mapped = {}
    for entry in entries:
        for name in entry.get("nuke", []):
            mapped.setdefault(name, set()).update(entry.get("nodebased", []))
    groups = knob_layout(node_type) if node_type else ()
    if node_type == "ViewMetaData":
        # This node's transient properties-panel controls do not live in SPECS/params.
        groups = (KnobGroup("table", ("metadata_table",), label="metadata viewer"),
                  KnobGroup("string", ("metadata_search",), label="search metadata for"),
                  KnobGroup("enum", ("metadata_filter_mode",), label="within"))
    splits = _split_table()["knobs"].get(key or "", {})
    row_sets = [set(g.params) for g in groups]
    components = {n for e in entries if e.get("kind") == "components" for n in e.get("nuke", [])}
    group_for = {param: g for g in groups for param in g.params}

    # Consecutive same-label entries define a shared guide row. Count every knob in such a row.
    row_total = row_match = 0
    row_missed, type_missed = [], []
    i = 0
    while i < len(knobs):
        j = i + 1
        while j < len(knobs) and knobs[j].get("label") == knobs[i].get("label") and str(knobs[i].get("label")).lower() not in ("n/a", ""):
            j += 1
        # A row counts the knobs NodeBased pairs; a Nuke knob with no NodeBased knob (the mask checkbox beside the mask channel) is
        # a missing knob, scored under `knobs`, and a row left with fewer than two paired knobs has no sharing to measure.
        block = [k for k in knobs[i:j] if mapped.get(k["name"])]
        if len(block) > 1:
            row_total += len(block)
            union = set().union(*(mapped[k["name"]] for k in block))
            if any(union <= row for row in row_sets):
                row_match += len(block)
            else:
                row_missed.append([k["name"] for k in block])
        i = j

    expected_counts = {"int": 0, "enum": 0, "bool": 0, "string": 0, "file": 0, "button": 0,
                       "float": 0, "number_list": 0, "channels": 0}
    type_match = type_total = unknown_count = 0
    read = node_type == "Read"
    viewer = node_type == "Viewer"
    per_item_names = {entry.get("nuke") for entry in pair.get("per_item", [])}
    for knob in knobs:
        name, nuke_kind = knob.get("name"), knob.get("type", "unknown")
        if name in per_item_names:
            continue
        if not node_type:
            if nuke_kind == "unknown":
                unknown_count += 1
            continue
        if read and name in _READ_KNOB_TYPES:
            nuke_kind = _READ_KNOB_TYPES[name]
        elif nuke_kind == "string" and str(knob.get("default")).strip().lower() in ("on", "off"):
            nuke_kind = "bool"      # the guide writes a checkbox's default as on or off
        elif nuke_kind == "number":
            nuke_kind = ("int" if (node_type, name) in _COUNT_KNOBS else "float" if (node_type, name) in _FLOAT_KNOBS
                         else "unknown" if (node_type, name) in _UNSURE_KNOBS else number_knob_kind(name, knob.get("default")))
        if nuke_kind == "unknown":
            unknown_count += 1
            continue
        if not knob.get("counted", True) or knob.get("role", "").startswith("chrome:"):
            continue
        if name not in mapped:
            continue
        params = mapped.get(name, set())
        # Empty `panel` pairs classified NOT_PIXEL (for example Tracker's viewer
        # presentation controls) count as paired controls, while having no node widget
        # whose type can be measured here.
        if not params and req.not_pixel_reason(node_type, name) and not viewer:
            continue
        actual = ([_READ_NODE_WIDGETS[p] for p in params if p in _READ_NODE_WIDGETS] if read
                  else [_VIEWER_WIDGETS[name]] if viewer and name in _VIEWER_WIDGETS
                  else ["enum" if is_open_choice(node_type, p) else _widget_kind(group_for[p], p) for p in params if p in group_for])
        if nuke_kind == "string" and not read and "enum" in actual:
            unknown_count += 1   # the guide infers `string` from any text default, and a menu's entry is text: it cannot tell the two apart
            continue
        if nuke_kind == "float" and (splits.get(name) or name in components or re.fullmatch(r"color\d*(#\d+)?", name)) and any(k in ("color", "split", "xy", "xyz") for k in actual):
            actual = ["float_slider"]   # an AColor or WH knob (docs/parity/nuke_split_knobs.json, or a knob named color) is one control that splits per channel
        expected_counts[nuke_kind] = expected_counts.get(nuke_kind, 0) + 1
        type_total += 1
        if read and name == "sequence":
            actual = ["string"]  # Read-only source readout on NodeBased' Sequence tab.
        if nuke_kind == "button" and read and name in ("reload", "updateLocalization"):
            matches = True  # Both are built as action buttons in the Read panel builder.
        elif nuke_kind == "int":
            matches = any(k == "int" for k in actual)
        elif nuke_kind == "float":
            matches = any(k in ("float_slider", "float", "split") for k in actual)
        elif nuke_kind == "enum" or nuke_kind == "channels":
            matches = any(k == "enum" for k in actual)
        elif nuke_kind == "bool":
            matches = any(k == "bool" for k in actual)
        elif nuke_kind == "file":
            matches = any(k in ("file", "file_read", "file_write", "string") for k in actual)
        elif nuke_kind == "string":
            matches = any(k in ("string", "multiline", "file_read", "file_write") for k in actual)
        elif nuke_kind == "number_list":
            matches = any(k in ("color", "xy", "xyz", "split") for k in actual)
        elif nuke_kind == "button":
            matches = any(k == "button" for k in actual)
        else:
            matches = False
        type_match += int(matches)
        if not matches:
            type_missed.append((name, nuke_kind, sorted(actual)))
    # Roto controls live on the selected payload item, not in knob_layout(). The
    # reviewable per_item records name the actual widget and its containing row;
    # item selection is part of this panel contract (the builders hide these
    # controls until a shape/stroke is selected).
    per_item = pair.get("per_item", [])
    # The guide records XY controls as `number`; for payload widgets, Nuke's
    # XY knob is the relevant widget class (two adjacent numeric fields).
    xy_knobs = {"translate", "scale", "source_transform_translate"}
    if node_type == "RotoPaint":
        xy_knobs.add("source_transform_scale")
    per_knobs = {k["name"]: k for k in knobs}
    per_row_match = per_row_total = per_type_match = per_type_total = 0
    per_row_missed, per_type_missed = [], []
    for item in per_item:
        knob = per_knobs.get(item["nuke"])
        if not knob or knob.get("type") == "unknown":
            continue
        kind = item.get("nuke_kind", knob.get("type"))
        if kind == "number":
            kind = ("number_list" if item["nuke"] in xy_knobs else
                    "int" if (node_type, item["nuke"]) in _COUNT_KNOBS else
                    "float" if (node_type, item["nuke"]) in _FLOAT_KNOBS else
                    "unknown" if (node_type, item["nuke"]) in _UNSURE_KNOBS else
                    number_knob_kind(item["nuke"], knob.get("default")))
        if kind == "unknown":
            continue
        actual_kind = item.get("widget_kind")
        per_type_total += 1
        if (kind == actual_kind or
                (kind == "float" and actual_kind in ("float", "float_slider", "color")) or
                (kind == "number_list" and actual_kind in ("color", "xy", "xyz", "split"))):
            per_type_match += 1
        else:
            per_type_missed.append((item["nuke"], kind, actual_kind))
        per_row_total += 1
        if item.get("row") and item.get("widget"):
            per_row_match += 1
        else:
            per_row_missed.append(item["nuke"])
    if per_row_total:
        row_match += per_row_match
        row_total += per_row_total
    if per_type_total:
        type_match += per_type_match
        type_total += per_type_total
        for missed in per_type_missed:
            type_missed.append(missed)
    return {"matched": row_match, "total": row_total, "score": row_match / row_total if row_total else None, "missed": row_missed + per_row_missed}, {
        "matched": type_match, "total": type_total, "score": type_match / type_total if type_total else None,
        "unknown": unknown_count, "unknown_names": [], "by_kind": expected_counts, "missed": type_missed,
    }


def summarize(units, field="score"):
    scored = [u for u in units if u[field] is not None]
    total_w = sum(u["weight"] for u in scored)
    overall = sum(u["weight"] * u[field] for u in scored) / total_w
    by_group = {}
    for g in GROUPS:
        gs = [u for u in scored if u["group"] == g]
        if gs:
            w = sum(u["weight"] for u in gs)
            by_group[g] = (sum(u["weight"] * u[field] for u in gs) / w, len(gs), sum(u[field] for u in gs) / len(gs))
    existing = [u for u in scored if u["nodebased"] is not None and not u.get("document")]
    w2 = sum(u["weight"] for u in existing)
    overall_existing = sum(u["weight"] * u[field] for u in existing) / w2
    return overall, overall_existing, by_group


# --------------------------------------------------------------------------------------------------------------- report
def fmt(x, digits=1):
    return "-" if x is None else f"{100 * x:.{digits}f}"


def pct(x):
    return "-" if x is None else f"{x:.1f}"


def gap_text(u):
    parts = []
    mp = u["detail"].get("missing_primary", [])
    if mp:
        parts.append("primary knobs missing: " + ", ".join(f"`{n}`" for n in mp[:8]) + (f" and {len(mp) - 8} more" if len(mp) > 8 else ""))
    ms = u["detail"].get("missing_secondary", [])
    if ms:
        parts.append(f"{len(ms)} secondary knob{'s' if len(ms) != 1 else ''} missing" + (": " + ", ".join(f"`{n}`" for n in ms[:5]) + (" ..." if len(ms) > 5 else "") if len(ms) <= 12 else ""))
    sk = u["detail"].get("missing_sockets", [])
    if sk:
        parts.append("sockets missing: " + ", ".join(f"`{n}`" for n in sk[:6]))
    mc = u["detail"].get("missing_chrome", [])
    if mc:
        parts.append("node chrome missing: " + ", ".join(CHROME_LABEL[r] for r in mc))
    mf = u["detail"].get("math_failed", [])
    if mf:
        parts.append("formula cases failing: " + ", ".join(f"`{n}`" for n in mf[:6]) + (f" and {len(mf) - 6} more" if len(mf) > 6 else ""))
    if u["detail"].get("math_unchecked"):
        parts.append("behaviour unchecked (no pixel case)")
    inert = u["detail"].get("inert_knobs", [])
    if inert:
        parts.append("knobs that change no pixels: " + ", ".join(f"`{n}`" for n in inert[:6]) + (f" and {len(inert) - 6} more" if len(inert) > 6 else ""))
    lo = u["detail"].get("layout_order")
    if lo and lo[0] < lo[1]:
        parts.append(f"knob order differs from Nuke's ({lo[0]} of {lo[1]} in order)")
    ll = u["detail"].get("layout_labels")
    if ll and ll[0] < ll[1]:
        parts.append(f"{ll[1] - ll[0]} label{'s differ' if ll[1] - ll[0] != 1 else ' differs'} from Nuke's")
    sm = u["detail"].get("split_missing", []) + u["detail"].get("split_partial", [])
    if sm:
        parts.append("not splittable like Nuke: " + ", ".join(f"`{n}`" for n in sm[:6]) + (f" and {len(sm) - 6} more" if len(sm) > 6 else ""))
    if u["nodebased"] is None:
        parts = [u.get("none_reason") or "no NodeBased node"]
    return "; ".join(parts) if parts else "nothing missing that the guide lists"


def build_report(reference_dir=None):
    nuke_doc, map_doc = load_json(NUKE_JSON), load_json(MAP_JSON)
    units, skipped = build_units(nuke_doc, map_doc, reference_dir)
    for u in units:
        if u["nodebased"] is None:
            u["none_reason"] = map_doc["pairs"][pair_key(u["group"], u["name"])].get("reason", "")
    overall, overall_existing, by_group = summarize(units)
    previous, previous_existing, previous_groups = summarize(units, "score_previous")
    layout_old, layout_old_existing, _ = summarize(units, "score_layout_old")
    unscored = [u for u in units if u["score"] is None]
    lines = []
    w = lines.append
    w("# 2D node fidelity against Nuke")
    w("")
    w(f"**Overall fidelity: {overall:.1f} / 100** ({len([u for u in units if u['score'] is not None])} nodes scored, rank-weighted; "
      f"{overall_existing:.1f} counting only the nodes NodeBased has).")
    w("")
    w(f"Under the previous formula (knob presence, defaults, sockets, chrome and formula cases only, used from 2026-10-25 to 2026-11-02) the same tree scores "
      f"{previous:.1f} ({previous_existing:.1f} counting only the nodes NodeBased has). The difference is the requirements added on 2026-11-02: knobs that act, "
      "Nuke's layout, per-channel splits, and behaviour that counts against a node until a pixel case checks it.")
    w("")
    w(f"Since 2026-11-04 the layout component is four equal quarters (order, labels, shared rows, widget types). Under the layout formula used from 2026-11-02 to 2026-11-04 "
      f"(order and labels as two halves) the same tree scores {layout_old:.1f} ({layout_old_existing:.1f} counting only the nodes NodeBased has).")
    w("")
    w("This is the score measured by `tools/audit_nuke_fidelity.py` (re-run it from a checkout; a test fails when this file drifts); the first measurement, step A on 2026-10-25, was 54.5. "
      "[PARITY_2D.md](PARITY_2D.md) answers *is there a node*; this file answers *how much of the real Nuke node is in it*. "
      "Every number is computed from the Nuke 17.0 Reference Guide's own tables, an explicit knob-by-knob mapping and formula tests on NodeBased's evaluator. "
      "Nothing here has been compared with a real Nuke render yet: the pixel-truth column is `unmeasured` until a folder of renders exists "
      "(`tools/make_nuke_reference_pack.py` writes the scripts that produce them).")
    w("")
    w("## How a node is scored")
    w("")
    w("Per node, up to nine components, each a fraction from 0 to 1. The parts marked *(11/2)* are the requirements added on 2026-11-02 "
      "(DiMo, 11/1: \"I need these nodes in a much better place\" and \"match known layouts and how some can hold multiple channel values\"):")
    w("")
    w("| Component | Weight | What it counts |")
    w("|---|---|---|")
    w(f"| knobs | {WEIGHTS['knobs']} | The Nuke node's knobs from the guide's tables, minus the shared chrome below, enable checkboxes, Python callbacks, GPU options and (for Read, Write, Viewer) per-format option tabs. Each knob NodeBased pairs with one counts; knobs on the node's main tab weigh 2, the rest 1. |")
    w(f"| acts | {WEIGHTS['acts']} | *(11/2)* Of the paired knobs, the share (main tab 2, the rest 1) that change NodeBased's output pixels when moved off their default on the fixture, "
      "through `Evaluator._kernel` (`tools/nuke_fidelity_requirements.py`). A knob that moves nothing alone is tried again with each other paired knob moved first. "
      "Knobs that act only over time, on metadata or on the bounding box are left out (listed below), and so are taps, whose behaviour is their input. Only nodes whose kernel runs on the fixture have this component. |")
    w(f"| defaults | {WEIGHTS['defaults']} | Of the paired knobs whose Nuke default the guide states and whose NodeBased default is comparable, the fraction that agree. |")
    w(f"| sockets | {WEIGHTS['sockets']} | The Nuke node's input sockets (mask included, A/B and numbered inputs as listed) that a NodeBased input is paired with. |")
    w(f"| chrome | {WEIGHTS['chrome']} | The shared node chrome the Nuke node has: channels selector, mask channel, mask invert, mask inject, mask fringe, (un)premult by, its invert, mix, bbox handling. |")
    w(f"| math | {WEIGHTS['math']} | Pixel tests of NodeBased's kernel against the behaviour the guide states (the 30 Merge operations, KeyMix, Gamma, Invert, Multiply, Add, Clamp, Premult, Unpremult, ColorMatrix) or Nuke's long-standing form, or a published measurement of Nuke itself, where the page names the knobs but states no formula (Grade, Exposure, ColorCorrect and the other grading nodes; marked *convention*). *(11/2)* A node whose kernel runs on the fixture and that has no case at all scores 0 here (behaviour unchecked); "
      "the previous formula left it out, which is how a node with no behaviour checked could score 100. A node whose every case is listed `not scored`, with its reason, and a tap, stay out. |")
    w(f"| layout | {WEIGHTS['layout']} | *(11/2, four quarters since 11/4)* Nuke's knob layout, the mean of four equal quarters. **Order**: the share of paired knobs that sit in the guide table's order in the Properties panel "
      "(the longest run already in order). **Labels**: the share of paired main-tab knobs whose label reads as the guide's label or the knob's scripting name (case and punctuation ignored; "
      "a label the guide gives a whole row of knobs, or `n/a`, is not compared). **Rows**: the share of paired knobs in a shared guide row (consecutive knobs with one label, such as Transform's translate x and y) that NodeBased places on one row; "
      "a Nuke knob with no NodeBased knob is not in the row, and a row left with fewer than two paired knobs is not measured. "
      "**Types**: the share of paired knobs of known guide type whose widget matches (integer field, slider with value field, dropdown, checkbox, text field, button, per-channel split). "
      "A quarter with nothing to count is left out and the rest renormalised. Section breaks are not measured. |")
    w(f"| splits | {WEIGHTS['splits']} | *(11/2)* For the Nuke knobs that split per channel (an AColor knob into r g b a, a WH knob into w h; "
      "`docs/parity/nuke_split_knobs.json`, see below), the share that are splittable NodeBased params (`core.SPLITTABLE`). A knob paired with one NodeBased param "
      "per channel and no one-slider form counts half. Only nodes the table lists have this component. |")
    w(f"| pixels | {WEIGHTS['pixels']} | Pixel truth against real Nuke renders. **Not measured yet**: it joins the formula once `--reference-dir` has renders; until then it is not in any number here. |")
    w("")
    w("`fidelity = 100 * sum(weight * fraction) / sum(weight)` over the components the node has. A component with nothing to count (the guide lists no inputs, no formula cases, no renders) is left out and the rest renormalised. "
      "A Nuke node with no NodeBased counterpart scores 0. A node whose guide page lists no inputs or knobs cannot be measured and is left out (listed at the end).")
    w("")
    w("Root is the one Nuke node scored against something that is not a node: the project settings. Its pairing (`document_settings` in `docs/parity/nodebased_to_nuke.json`) names document fields, such as `time.first` or `settings.color.view`, where other pairings name a node's params. "
      "A Root knob counts as covered only when its document field exists and changes behaviour (`tests/test_2d_parity_notimeblur.py`, `RootPairingTests`, shows each one acting); every other Root knob is listed there as unsupported with a one-line reason and scores as missing. "
      "The defaults component compares a paired field with the value a new project starts with (`core.demo_document`). Root has no input sockets, shared chrome or formula cases, so those components are left out and only knobs and defaults are renormalised.")
    w("")
    w("**Overall** is the mean of node fidelity weighted by daily-use rank: a node in row *r* of its group weighs `1 / r`, shared equally by the Nuke nodes of a combined row "
      "(`Grain / ScannedGrain` gives each half of `1 / 9`). Rows marked n/a in PARITY_2D.md are not counted.")
    w("")
    w("## By group")
    w("")
    w("| Group | Rank-weighted | Plain mean | Previous formula | Nodes |")
    w("|---|---|---|---|---|")
    for g in GROUPS:
        if g in by_group:
            rw, n, mean = by_group[g]
            w(f"| {g} | {rw:.1f} | {mean:.1f} | {previous_groups[g][0]:.1f} | {n} |")
    w("")
    checked = [u for u in units if u["score"] is not None and "math" in u["components"]]
    scored_nodes = [u for u in units if u["score"] is not None]
    checked = [u for u in checked if not u["detail"].get("math_unchecked")]
    unchecked = [u for u in scored_nodes if u["detail"].get("math_unchecked")]
    probed = sum(u["detail"].get("acts_counts", (0, 0))[1] for u in scored_nodes)
    acting = sum(u["detail"].get("acts_counts", (0, 0))[0] for u in scored_nodes)
    split_nodes = [u for u in scored_nodes if "splits" in u["components"]]
    w(f"Behaviour checked: {len(checked)} of the {len(scored_nodes)} scored nodes have at least one scored pixel case against what the node computes "
      f"(the `math` column below); {len(unchecked)} nodes whose kernel runs on the fixture have none and score 0 on `math`.")
    w("")
    w(f"Knobs that act: {acting} of the {probed} paired knobs probed change the output pixels; "
      f"{probed - acting} change nothing on the fixture (listed per node below). "
      f"Splits: {sum(1 for u in split_nodes if u['components']['splits'] == 1)} of the {len(split_nodes)} nodes with split knobs in the table split all of them.")
    w("")
    w("## Per node")
    w("")
    w("Columns: the Nuke node, the NodeBased kind paired with it, the row's PARITY_2D status, fidelity, then the component fractions as percentages "
      "(`knobs` with the count of Nuke knobs covered, `acts` with acting/probed knobs, `defaults`, `sockets`, `chrome`, `math` with passed/total cases, "
      "`unchecked` when no case exists, `layout`, `splits` with fully split/listed knobs) and the pixel-truth column. `Prev.` is the node under the previous formula.")
    for g in GROUPS:
        gu = [u for u in units if u["group"] == g]
        if not gu:
            continue
        w("")
        w(f"### {g}")
        w("")
        w("| Rank | Nuke node | NodeBased | Row status | Fidelity | Prev. | knobs | acts | defaults | sockets | chrome | math | layout | splits | pixels |")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for u in sorted(gu, key=lambda x: (x["rank"], x["name"])):
            c, d = u["components"], u["detail"]
            kc = f" ({d['knob_counts'][0]}/{d['knob_counts'][1]})" if "knob_counts" in d else ""
            mc = f" ({d['math'][0]}/{d['math'][1]})" if "math" in d else (" unchecked" if d.get("math_unchecked") else "")
            ac = f" ({d['acts_counts'][0]}/{d['acts_counts'][1]})" if "acts_counts" in d and "acts" in c else ""
            sc = f" ({d['split_counts'][0]}/{d['split_counts'][1]})" if "split_counts" in d else ""
            w(f"| {u['rank']} | {u['name']} | {'`' + u['nodebased'] + '`' if u['nodebased'] else '-'} | {u['status']} | "
              f"{pct(u['score']) if u['score'] is not None else 'n/a'} | {pct(u['score_previous']) if u['score_previous'] is not None else 'n/a'} | "
              f"{fmt(c.get('knobs'), 0)}{kc} | {fmt(c.get('acts'), 0)}{ac} | {fmt(c.get('defaults'), 0)} | "
              f"{fmt(c.get('sockets'), 0)} | {fmt(c.get('chrome'), 0)} | {fmt(c.get('math'), 0)}{mc} | {fmt(c.get('layout'), 0)} | "
              f"{fmt(c.get('splits'), 0)}{sc} | {u['pixel_column']} |")
    w("")
    w("## Panel rows and widget types (two quarters of the layout component; 2026-11-04)")
    w("")
    w("Rows count paired knobs in consecutive same-label groups from the guide that NodeBased places together on one panel row (a guide knob with no NodeBased knob is not counted, a row of fewer than two paired knobs is not measured, and the label `N/A` shares nothing). "
      "Types count paired knobs with a known guide type whose widget matches. The guide's `type` is inferred from defaults, so a `number` is resolved as follows: "
      "an integer field with no slider when the guide default is an integer and the knob's name counts, indexes or names a frame (frame, first, last, count, samples, iterations, index, seed, octaves, divisions, steps, levels, interval, period and the like, and the few knobs listed by node because their name does not say so: Tracker's smoothT, smoothR and smoothS, which average over that many frames, and Switch's which, which picks an input by index); "
      "otherwise a float with a slider and value field (Nuke's Double_Knob), which a per-channel split, or a colour, xy or xyz row for a knob named color, listed in the split table, or paired by components onto such a row (Glow's tint, Keylight's edge color, Reformat's box width and height), also satisfies; with no numeric default the type is left unmeasured. "
      "The Viewer's knobs are viewer state rather than node params, so each is measured against the control it has in the Viewer settings window or the viewer strip (a fixed table in the tool, checked against the real widgets by tests.test_panel_parity_sweep). "
      "A `string` paired with a dropdown is left unmeasured (the guide gives a menu's current entry as text and cannot tell the two apart), and a `string` defaulting to on or off is a checkbox. "
      "A `number_list` needs a colour, xy, xyz or split row; `enum` and `channels` need a dropdown; `bool` a checkbox. Unmeasured knobs are counted in the last column and never score.")
    w("| Node | Shared-row knobs on one row | Widget types matching | Unknown types (unmeasured) |")
    w("|---|---:|---:|---:|")
    for u in sorted(units, key=lambda x: (GROUPS.index(x["group"]), x["rank"], x["name"])):
        rows, types = u["panel_rows"], u["panel_types"]
        row_text = f"{rows['score'] * 100:.1f}% ({rows['matched']}/{rows['total']})" if rows["total"] else "—"
        type_text = f"{types['score'] * 100:.1f}% ({types['matched']}/{types['total']})" if types["total"] else "—"
        w(f"| {u['name']} | {row_text} | {type_text} | {types['unknown']} |")
    w("")
    sweep = {u["name"]: u for u in units}
    w("Panel sweep S2 (2026-11-04) baseline for its eight nodes, before rows were placed on one row, counting knobs were integer fields and the Viewer's controls were measured: "
      "Tracker rows 3/5, types 27/34; Viewer types 0/20; Switch types 0/1; TimeClip rows 0/6; FrameRange rows 0/2; Glow types 8/9; Keylight types 33/34; Reformat types 12/14. "
      "Current: " + "; ".join(
          f"{n} rows {sweep[n]['panel_rows']['matched']}/{sweep[n]['panel_rows']['total']}, types {sweep[n]['panel_types']['matched']}/{sweep[n]['panel_types']['total']}"
          for n in ("Tracker", "Viewer", "Switch", "TimeClip", "FrameRange", "Glow", "Keylight", "Reformat") if n in sweep) + ". "
      "Tracker's `motionblur` (a number in the guide, an on/off switch here) and `cornerPinOptions` (a menu in the guide, two export buttons here), and the Viewer's `clip_warning` (a menu in the guide, the zebra check box here), "
      "`downrez` (a number in the guide, the proxy-tier menu here) and `overscan` (a number in the guide, a whole-pixel field here) stay as they are: each would need a change to how the node renders or to viewer state.")
    w("")
    w("Panel sweep S3 (2026-11-04) baseline for its eight nodes, before rows were placed on one row and the Blend check box was paired with the guide's `weight1` (the guide's table names the normalize check box that): "
      "Remove rows 0/3; AppendClip types 4/5; CornerPin2D rows 0/2, types 10/12; BurnIn rows 0/2; Clamp rows 0/8; Blend types 1/2; DirBlur types 9/11; ContactSheet rows 0/6, types 8/11. "
      "Current: " + "; ".join(
          f"{n} rows {sweep[n]['panel_rows']['matched']}/{sweep[n]['panel_rows']['total']}, types {sweep[n]['panel_types']['matched']}/{sweep[n]['panel_types']['total']}"
          for n in ("Remove", "AppendClip", "CornerPin2D", "BurnIn", "Clamp", "Blend", "DirBlur", "ContactSheet") if n in sweep) + ". "
      "CornerPin2D's `invert` (a check box in the guide) is a check box over the stored forward/inverse text, DirBlur's `target` (an xy pair) is an x, y pair of fields over the stored text \"x,y\", "
      "and DirBlur's `Quality` (a number) is a whole-number field with a slider over the stored low/normal/high (1, 2 and 4; any other whole number is stored as its digits). A saved document keeps the text it held and renders as before. "
      "CornerPin2D's `motionblur` (a number in the guide, an on/off switch beside a samples count here) stays as it is, as Tracker's and Transform's do: one number would change how the node renders. "
      "AppendClip's `dissolve` (the length of the cross-fade in frames) and ContactSheet's `width`, `height` and `gap` (whole pixels) are listed as counting knobs. Clamp's enable check boxes sit beside the per-channel value they switch on, with the split button still on the value.")
    w("")
    w("Panel sweep S4 (2026-11-04) baseline for its nodes, before rows were placed on one row and the count and fraction knobs were listed: "
      "TimeBlur rows 0/2; TimeEcho types 2/3; ZDefocus types 24/25; IBKGizmo types 13/14; Encryptomatte rows 0/2; Kronos rows 0/2, types 12/14; "
      "IBKColor types 1/5; OFlow rows 0/2, types 10/12; CopyRectangle rows 0/4; TimeDissolve types 0/2; RotoPaint types 9/62. "
      "Current: " + "; ".join(
          f"{n} rows {sweep[n]['panel_rows']['matched']}/{sweep[n]['panel_rows']['total']}, types {sweep[n]['panel_types']['matched']}/{sweep[n]['panel_types']['total']}"
          for n in ("TimeBlur", "TimeEcho", "ZDefocus", "IBKGizmo", "Encryptomatte", "Kronos", "IBKColor", "OFlow", "CopyRectangle", "TimeDissolve", "RotoPaint") if n in sweep) + ". "
      "Roto and RotoPaint per-item rows and widget types are measured through the `per_item` payload-to-selected-control mappings. "
      "Roto currently measures rows 40/40 and types 55/56; RotoPaint rows 51/53 and types 64/66. The two remaining RotoPaint gaps are dynamic size and transparency: "
      "the current payload has no corresponding fields or controls. `global_motionblur` remains a node-level samples count beside an on/off switch. "
      "Kronos's and OFlow's `vectorDetail` (a fraction from 0 to 1 in the guide, a whole number of 1 to 6 pyramid levels here) and IBKColor's `off` and `mult` (RGB triples in the guide, single numbers here) "
      "stay: each needs a change to the kernel. IBKColor's `Size` and `multi` are left unmeasured because the guide's defaults cannot tell the widget. "
      "TimeEcho's frames behind, ZDefocus's blades and TimeDissolve's in and out are listed as counting knobs; IBKGizmo's screen range and the Kronos and OFlow timing frame are listed as fractions.")
    w("")
    s5 = ("Grid", "Colorspace", "MotionBlur2D", "MotionBlur3D", "SmartVector", "VectorBlur", "OCIODisplay", "OCIOFileTransform", "OCIOLookTransform", "Sparkles")
    w("Panel sweep S5 (2026-11-07) baseline for its nodes, before rows were placed on one row and the count knobs were listed: "
      "Grid types 11/12; Colorspace rows 0/6; MotionBlur2D and MotionBlur3D rows 0/2; SmartVector types 7/9; VectorBlur types 10/12; "
      "OCIODisplay types 6/7; OCIOFileTransform types 2/3; OCIOLookTransform types 3/4; Sparkles types 26/27. "
      "Current: " + "; ".join(
          f"{n} rows {sweep[n]['panel_rows']['matched']}/{sweep[n]['panel_rows']['total']}, types {sweep[n]['panel_types']['matched']}/{sweep[n]['panel_types']['total']}"
          for n in s5 if n in sweep) + ". "
      "Grid's Number is one row of two whole-number fields; Colorspace's Input and Output are one row each of three menus; MotionBlur's shutter offset menu and custom offset share a row. "
      "OCIODisplay's, OCIOFileTransform's and OCIOLookTransform's Invert direction is a check box over the stored direction text (forward or inverse), as CornerPin2D's Invert is. "
      "SmartVector's matteDilation, VectorBlur's grow_bbox and Sparkles's ray_nb are listed as counting knobs. "
      "Root is the Project Settings dialog, not a node panel, and the ruler reads node panels only, so its frame-range and proxy rows are not measured here. "
      "SmartVector's `vectorDetailReg` (a fraction in the guide, an integer 1 to 6 here) needs a change to the kernel, and that stays the one types mismatch on SmartVector. VectorBlur's `uv` (a layer picker in the guide) is an editable menu of the motion and SmartVector layer names over the same stored layer name (panel sweep S5 finish); STMap and IDistort keep their layer name field.")
    w("")
    read_unit = next((u for u in units if u["name"] == "Read"), None)
    if read_unit:
        r, t = read_unit["panel_rows"], read_unit["panel_types"]
        w(f"Read baseline before Lane 9 step R1: shared rows 0/8 (0.0%) and widget types 25/29 (86.2%); "
          "the reload counter and three Boolean toggles were the four mismatches. Current measurements are "
          f"rows {r['matched']}/{r['total']} ({r['score'] * 100:.1f}%) and types {t['matched']}/{t['total']} ({t['score'] * 100:.1f}%). "
          f"Read fidelity is unchanged at {read_unit['score']:.1f}.")
        w("")
    for title, metric in (("Lowest shared-row scores", "panel_rows"), ("Lowest widget-type scores", "panel_types")):
        ranked_shape = sorted((u for u in units if u[metric]["score"] is not None),
                              key=lambda u: (u[metric]["score"], u["name"]))[:5]
        w(f"**{title}:** " + ("; ".join(f"{u['name']} ({u[metric]['score'] * 100:.1f}%)" for u in ranked_shape) or "none measurable") + ".")
    w("")
    w(f"## The {GAP_LIST} largest gaps")
    w("")
    w("Ranked by `rank weight * (100 - fidelity)`, the share of the overall shortfall each node accounts for. This is the backlog for the following steps.")
    w("")
    w("| # | Nuke node | NodeBased | Weighted gap | Fidelity | What is missing |")
    w("|---|---|---|---|---|---|")
    ranked = sorted([u for u in units if u["score"] is not None], key=lambda u: (-u["weight"] * (100 - u["score"]), u["rank"], u["name"]))
    for i, u in enumerate(ranked[:GAP_LIST], 1):
        w(f"| {i} | {u['name']} ({u['group']} #{u['rank']}) | {'`' + u['nodebased'] + '`' if u['nodebased'] else '-'} | "
          f"{u['weight'] * (100 - u['score']):.2f} | {u['score']:.1f} | {gap_text(u)} |")
    w("")
    w("## Formula cases")
    w("")
    w("Each case runs one node on the 16 x 12 float fixture (HDR values from -0.5 to 12, alpha from 0 to 1) through `Evaluator._kernel` and compares every pixel with the stated behaviour "
      f"(absolute and relative tolerance {ALLCLOSE['atol']:g}, except the motion-estimated cases of Kronos and OFlow, which state an absolute tolerance of 0.01 in their statements because an estimated field is not exact; a failing case shows its largest error rounded to four decimals and the first pixel in raster order that has it). `guide` means the guide's text states the behaviour; `convention` means it does not (see the ambiguities below). "
      "`inexpressible` means NodeBased has no knob that can set what Nuke's knob sets, so it cannot even attempt the case.")
    w("")
    w("| Case | Source | Result | Max abs error | Worst pixel (y, x, channel) |")
    w("|---|---|---|---|---|")
    from nuke_fidelity_cases import CASES
    for case in CASES:
        r = run_case(case)
        status = r["status"] + (": " + r["unscored"] if r["unscored"] else "")
        failed = r["status"] == "fail"
        err = f"{r['max_abs_error']:.3g}" if failed else ("within tolerance" if r["status"] == "pass" else "-")
        worst = str(tuple(r["worst"])) if failed else "-"
        w(f"| `{case.id}` | {case.source} | {status} | {err} | {worst} |")
    w("")
    w("## Requirements of 2026-11-02, per node")
    w("")
    w("### Knobs that change no pixels")
    w("")
    w("Each paired knob below was moved off its default to every value its `LIMITS` and `CHOICES` allow the probe to try, alone and with each other paired knob moved first, "
      "and NodeBased's output did not change on the fixture. Either the knob is not implemented, or it acts only in a state the probe does not reach "
      "(two other knobs moved together, a payload the default node does not hold). Implement it, or unpair it in `docs/parity/nodebased_to_nuke.json` with a reason.")
    w("")
    import nuke_fidelity_requirements as req
    for u in [u for u in units if u["detail"].get("inert_knobs")]:
        w(f"- {u['name']} (`{u['nodebased']}`): " + ", ".join(f"`{n}`" for n in u["detail"]["inert_knobs"]))
    w("")
    w("Not probed, because a one-frame pixel probe cannot see them: " + "; ".join(
        f"{why}: " + ", ".join(f"`{n}`" for n in sorted(k for k, r in req.NOT_PIXEL["*"].items() if r == why))
        for why in sorted(set(req.NOT_PIXEL["*"].values()))) + "; " + "; ".join(
        f"{node}'s `{k}` ({r})" for node, ks in sorted(req.NOT_PIXEL.items()) if node != "*" for k, r in ks.items()) + ". "
      "Whole nodes not probed: " + "; ".join(f"{n} ({r})" for n, r in sorted(req.NOT_PROBED.items())) + ". "
      "Taps, whose behaviour is their input and which have neither `acts` nor `math`: " + ", ".join(f"`{t}`" for t in req.TAPS) + ".")
    w("")
    w("### Layout differences")
    w("")
    w("Labels below are NodeBased's, then the guide's.")
    w("")
    for u in [u for u in units if u["detail"].get("label_mismatches") or (u["detail"].get("layout_order") and u["detail"]["layout_order"][0] < u["detail"]["layout_order"][1])]:
        lo = u["detail"].get("layout_order")
        bits = []
        if lo and lo[0] < lo[1]:
            bits.append(f"{lo[0]} of {lo[1]} paired knobs in Nuke's order")
        bits += [f"`{n}` reads \"{ours}\", Nuke \"{theirs}\"" for n, ours, theirs in u["detail"].get("label_mismatches", [])]
        w(f"- {u['name']} (`{u['nodebased']}`): " + "; ".join(bits))
    w("")
    w("### Split knobs")
    w("")
    table = _split_table()
    w(table["source"] + " " + table["status"])
    w("")
    for u in [u for u in units if "splits" in u["components"]]:
        d = u["detail"]
        state = "all split" if u["components"]["splits"] == 1 else ", ".join(
            [f"`{n}` not splittable" for n in d.get("split_missing", [])] + [f"`{n}` per-channel params, no one-slider form" for n in d.get("split_partial", [])])
        w(f"- {u['name']} (`{u['nodebased'] or '-'}`): {', '.join(f'`{k}` ({v})' for k, v in table['knobs'][u['nuke']].items())}. {state}.")
    w("")
    w("## What a reader should know about the reference data")
    w("")
    for line in AMBIGUITIES:
        w(f"- {line}")
    w("")
    w("### Guide pages without a row in PARITY_2D.md")
    w("")
    covered = {(n["group"], n["page"].rsplit("/", 1)[1][:-5]) for n in nuke_doc["nodes"].values() if n.get("page")}
    index = nuke_doc.get("guide_index", {})
    extra = [f"{g}/{slug}" for g in GROUPS for slug in index.get(g, []) if (g, slug) not in covered]
    w("The guide's group pages list these nodes that no row of PARITY_2D.md covers, so this score cannot see them: " + ", ".join(f"`{e}`" for e in extra) + ".")
    w("")
    w("### Nodes not scored")
    w("")
    if unscored:
        for u in unscored:
            w(f"- {u['name']} ({u['group']} #{u['rank']}): the guide page lists no inputs and no knobs.")
    for g, r, n, why in skipped:
        w(f"- {n} ({g} #{r}): {why}.")
    w("")
    w("### NodeBased 2D kinds paired with no Nuke node")
    w("")
    from nodebased.nodecatalog import NODE_CATEGORIES
    paired = {p["nodebased"] for p in map_doc["pairs"].values() if p.get("nodebased")}
    orphans = [k for g in GROUPS for k in NODE_CATEGORIES[g] if k not in paired]
    w("These kinds are in the node catalogue's Image to Other groups but are not scored, because no node in PARITY_2D.md pairs with them: "
      + ", ".join(f"`{k}`" for k in orphans) + ".")
    w("")
    return "\n".join(lines) + "\n"


AMBIGUITIES = [
    "The guide has no type column. A knob's `type` in `docs/parity/nuke_nodes.json` is inferred from its default (`enabled`/`disabled` is a bool, a number is a number, several numbers a list) and from the choices the text lists.",
    "Where a default reads 'Dependent on ...', 'dependent on file' or is prose, the data records `null` with a note. These knobs count as covered or missing like any other but never enter the defaults component.",
    "The guide does not say how a colour or position knob is stored. NodeBased splits them into scalar params (`red`, `green`, `blue`, `alpha`; `p0_x`, `p0_y`); the mapping file lists them as `components` of one Nuke knob.",
    "Multiply's `value` is listed with default 0, which would blank the image; Nuke's own default is 1. The score skips that default (`skip_default`).",
    "Exposure's `mode` default is listed as `Densities`; the long-standing Nuke default is `Stops`. NodeBased defaults to stops, so the defaults component counts it as a mismatch.",
    "The guide spells Sharpen's, Soften's and Laplacian's quality knob `quailty`; the data keeps the guide's spelling.",
    "Clamp's `MinClampTo_ enable` has a stray space in the guide's table; the data removes whitespace from knob names (`MinClampTo_enable`).",
    "The guide repeats scripting names inside one node (`invert`, `output`, `color`, `area`): the data suffixes the repeats in table order (`area#2`, `area#3`, `area#4` are the y, r and t halves of CopyRectangle's area).",
    "No guide page exists for PlanarTracker, and the Stabilize and Backdrop pages list no inputs or knobs; those nodes are not scored.",
    "Gamma is described as 'the actual function is 1/pow(value)'. That sentence is not a function of the pixel; the formula case reads it as `out = in ** (1 / value)` and tests only non-negative input.",
    "The guide states no formula for Grade or Exposure. Their cases use Nuke's long-standing forms and are marked *convention*; the lane brief's `A*(in-B)^gamma` shape is the Grade form.",
    "The Merge page's algorithm for `matte` ('screen(A,a,B,b) if A==a, B if a<=0, ...') cannot be implemented without guessing, so that case is listed and not scored. `soft-light`, `hard-light`, `screen`, `multiply` and `divide` are implemented literally from the guide's text, "
    "which is not always what Nuke does outside 0..1; a failure there is a difference from the guide as written.",
    "The guide's mask input is listed as a socket only on some pages whose nodes do have a mask channel knob (for example Crop has neither). The sockets component counts what each page lists.",
    "Keyer's `range` knob is listed in the guide's table with 1 for every point; Nuke's own range starts at 0 0 1 1 (A, B, C, D), which is what the node has, so its first two points are skipped by the defaults component (`skip_default`).",
    "A text default the guide lists as `none` is an empty field in NodeBased (Read and Write `proxy`, `file`, `frame`, the metadata `key` and `value` rows, Write's `file_type` Auto); the mapping's `value_map` reads the empty value as `none`. Read's `proxy_format` starts at `file` (the proxy takes its own size) where the guide lists 1024x778, a size that depends on the project, so it is skipped.",
    "Defaults changed to the guide's in 2D behaviour step P (a new node only; a saved document keeps the older value through schema 32, `nodebased/olderdefaults.py`): AddTimeCode's start code (01:00:00:00), TimeEcho's frames behind (1) and mode (max), TimeBlur's shutter (0.5) and shutter offset (start), and ContactSheet's width (3072), height (2048), rows (3) and gap (0). ContactSheet's `startframe` and `endframe` keep 1 where the guide lists -1: they act only with Split inputs on, where a frame before the first would read nothing. "
    "Defaults NodeBased keeps on purpose (2D behaviour step M): BurnIn's text slots, which hold `[metadata input/filename]`, `[frame]` and `[metadata timecode]` where the guide lists the Hiero expressions `hiero/clip`, `hiero/shot`, `hiero/track` and `hiero/sequence/timecode` (NodeBased has no Hiero metadata), and its `fontfamily`, empty where the guide lists Utopia (the system font stands in); Write's `create_directories`, on, so a render into a new folder does not fail where the guide lists off. A new Read starts with `auto_alpha` off and a new Write on `channels` rgb, as the guide lists (`readknobs.NEW_NODE_DEFAULTS`, `writenode.NEW_NODE_DEFAULTS`); a saved Read or Write that lacks the knob reads with the value of before (`readknobs.DEFAULTS`, `writenode.DEFAULTS`: auto alpha on, every channel), so older documents render and write as they did. The audit lists all of these as differences.",
    "Features NodeBased keeps in a payload or panel rather than in a node param (Roto and RotoPaint shape and stroke fields, the Viewer's look and proxy state, the metadata viewers) are paired as `panel` entries, each citing the document that describes it; they count as covered. A Nuke knob NodeBased cannot honour is paired as `unsupported` with the reason; it counts as missing, so listing it explains a gap without closing it. Roto's CornerPin export tab is not paired.",
]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true", help="rewrite docs/PARITY_2D_FIDELITY.md")
    ap.add_argument("--check", action="store_true", help="exit 1 when docs/PARITY_2D_FIDELITY.md differs from a fresh run")
    ap.add_argument("--reference-dir", help="folder of Nuke renders (see tools/make_nuke_reference_pack.py) to score pixel truth against")
    ap.add_argument("--summary", action="store_true", help="print the overall score and the group means only")
    args = ap.parse_args(argv)
    text = build_report(args.reference_dir)
    if args.write:
        REPORT.write_text(text, encoding="utf-8")
        print(f"wrote {REPORT}")
        return 0
    if args.check:
        if not REPORT.exists() or REPORT.read_text(encoding="utf-8") != text:
            print("docs/PARITY_2D_FIDELITY.md differs from a fresh run; re-run with --write", file=sys.stderr)
            return 1
        print("docs/PARITY_2D_FIDELITY.md is current")
        return 0
    if args.summary:
        print("\n".join(text.splitlines()[:3]))
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
