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
WEIGHTS = {"knobs": 40, "defaults": 10, "sockets": 10, "chrome": 15, "math": 25, "pixels": 30}
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
            tol = ALLCLOSE["atol"] + ALLCLOSE["rtol"] * np.abs(ref)
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
    from nuke_fidelity_cases import cases_for
    res = {"nuke": key, "nodebased": pair.get("nodebased") or pair.get("document_settings"), "components": {}, "detail": {}}
    knobs = [k for k in nuke["knobs"] if k["counted"] and not k["role"].startswith("chrome:")]
    chrome = {}
    for k in nuke["knobs"]:
        if k["role"].startswith("chrome:"):
            chrome.setdefault(k["role"][7:], []).append(k["name"])
    sockets = list(dict.fromkeys(i["name"] for i in nuke["inputs"]))
    cases = [c for c in cases_for(key) if not c.unscored]
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
    # documented math
    if cases:
        results = [run_case(c) for c in cases]
        passed = sum(1 for r in results if r["status"] == "pass")
        res["components"]["math"] = passed / len(results)
        res["detail"]["math"] = (passed, len(results))
        res["detail"]["math_failed"] = [r["id"] for r in results if r["status"] != "pass"]
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
    res["document"] = document
    res["pixel_column"] = ("measured" if truth else ("unmeasured" if cases_for(key) else "no fixture"))
    return res


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
            res.update(group=group, rank=rank, name=n, status=status, weight=(1.0 / rank) / len(members))
            units.append(res)
    return units, skipped


def summarize(units):
    scored = [u for u in units if u["score"] is not None]
    total_w = sum(u["weight"] for u in scored)
    overall = sum(u["weight"] * u["score"] for u in scored) / total_w
    by_group = {}
    for g in GROUPS:
        gs = [u for u in scored if u["group"] == g]
        if gs:
            w = sum(u["weight"] for u in gs)
            by_group[g] = (sum(u["weight"] * u["score"] for u in gs) / w, len(gs), sum(u["score"] for u in gs) / len(gs))
    existing = [u for u in scored if u["nodebased"] is not None and not u.get("document")]
    w2 = sum(u["weight"] for u in existing)
    overall_existing = sum(u["weight"] * u["score"] for u in existing) / w2
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
    unscored = [u for u in units if u["score"] is None]
    lines = []
    w = lines.append
    w("# 2D node fidelity against Nuke")
    w("")
    w(f"**Overall fidelity: {overall:.1f} / 100** ({len([u for u in units if u['score'] is not None])} nodes scored, rank-weighted; "
      f"{overall_existing:.1f} counting only the nodes NodeBased has).")
    w("")
    w("This is the score measured by `tools/audit_nuke_fidelity.py` (re-run it from a checkout; a test fails when this file drifts); the first measurement, step A on 2026-10-25, was 54.5. "
      "[PARITY_2D.md](PARITY_2D.md) answers *is there a node*; this file answers *how much of the real Nuke node is in it*. "
      "Every number is computed from the Nuke 17.0 Reference Guide's own tables, an explicit knob-by-knob mapping and formula tests on NodeBased's evaluator. "
      "Nothing here has been compared with a real Nuke render yet: the pixel-truth column is `unmeasured` until a folder of renders exists "
      "(`tools/make_nuke_reference_pack.py` writes the scripts that produce them).")
    w("")
    w("## How a node is scored")
    w("")
    w("Per node, five components, each a fraction from 0 to 1:")
    w("")
    w("| Component | Weight | What it counts |")
    w("|---|---|---|")
    w("| knobs | 40 | The Nuke node's knobs from the guide's tables, minus the shared chrome below, enable checkboxes, Python callbacks, GPU options and (for Read, Write, Viewer) per-format option tabs. Each knob NodeBased pairs with one counts; knobs on the node's main tab weigh 2, the rest 1. |")
    w("| defaults | 10 | Of the paired knobs whose Nuke default the guide states and whose NodeBased default is comparable, the fraction that agree. |")
    w("| sockets | 10 | The Nuke node's input sockets (mask included, A/B and numbered inputs as listed) that a NodeBased input is paired with. |")
    w("| chrome | 15 | The shared node chrome the Nuke node has: channels selector, mask channel, mask invert, mask inject, mask fringe, (un)premult by, its invert, mix, bbox handling. |")
    w("| math | 25 | Pixel tests of NodeBased's kernel against the behaviour the guide states (the 30 Merge operations, KeyMix, Gamma, Invert, Multiply, Add, Clamp, Premult, Unpremult, ColorMatrix) or Nuke's long-standing form where the page names the knobs but states no formula (Grade, Exposure; marked *convention*). Only nodes with such cases have this component. |")
    w("| pixels | 30 | Pixel truth against real Nuke renders. **Not measured yet**: it joins the formula once `--reference-dir` has renders; until then it is not in any number here. |")
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
    w("| Group | Rank-weighted | Plain mean | Nodes |")
    w("|---|---|---|---|")
    for g in GROUPS:
        if g in by_group:
            rw, n, mean = by_group[g]
            w(f"| {g} | {rw:.1f} | {mean:.1f} | {n} |")
    w("")
    w("## Per node")
    w("")
    w("Columns: the Nuke node, the NodeBased kind paired with it, the row's PARITY_2D status, fidelity, then the component fractions as percentages "
      "(`knobs` with the count of Nuke knobs covered, `defaults`, `sockets`, `chrome`, `math` with passed/total cases) and the pixel-truth column.")
    for g in GROUPS:
        gu = [u for u in units if u["group"] == g]
        if not gu:
            continue
        w("")
        w(f"### {g}")
        w("")
        w("| Rank | Nuke node | NodeBased | Row status | Fidelity | knobs | defaults | sockets | chrome | math | pixels |")
        w("|---|---|---|---|---|---|---|---|---|---|---|")
        for u in sorted(gu, key=lambda x: (x["rank"], x["name"])):
            c, d = u["components"], u["detail"]
            kc = f" ({d['knob_counts'][0]}/{d['knob_counts'][1]})" if "knob_counts" in d else ""
            mc = f" ({d['math'][0]}/{d['math'][1]})" if "math" in d else ""
            w(f"| {u['rank']} | {u['name']} | {'`' + u['nodebased'] + '`' if u['nodebased'] else '-'} | {u['status']} | "
              f"{pct(u['score']) if u['score'] is not None else 'n/a'} | {fmt(c.get('knobs'), 0)}{kc} | {fmt(c.get('defaults'), 0)} | "
              f"{fmt(c.get('sockets'), 0)} | {fmt(c.get('chrome'), 0)} | {fmt(c.get('math'), 0)}{mc} | {u['pixel_column']} |")
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
      f"(absolute and relative tolerance {ALLCLOSE['atol']:g}; a failing case shows its largest error rounded to four decimals and the first pixel in raster order that has it). `guide` means the guide's text states the behaviour; `convention` means it does not (see the ambiguities below). "
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
