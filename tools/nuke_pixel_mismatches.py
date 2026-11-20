#!/usr/bin/env python3
"""Sort the cases where NodeBased's pixels differ from Nuke's own render, by the lane that owns the node.

    python tools/nuke_pixel_mismatches.py --reference-dir RENDER_DIR      # writes docs/parity/review/nuke_pixels.json

For every case with a render it records NodeBased against Nuke (the largest error, its channel and worst pixel, row first) and
Nuke against the result the guide states (the case's expectation, over the pixels the expectation covers). A case where NodeBased
meets the guide but Nuke does not is the most telling row: there Nuke disagrees with what its guide says, and the guide was the only
reference until renders existed. The JSON holds numbers only, never pixels, so it can be committed while the renders stay local;
`tools/parity_review_book.py --worklist` lists it in WORKLIST.md.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

OUT = ROOT / "docs" / "parity" / "review" / "nuke_pixels.json"
CHANNELS = "RGBA"
TOL = 1e-4                                         # make_nuke_reference_pack.TOLERANCE: atol and rtol


def _worst(err, theirs, mask=None):
    """(largest error, channel letter, [row, column]) over `mask`; NaN errors (a render without alpha) are left out."""
    err = np.where(np.isfinite(err), err, 0.0)
    if mask is not None:
        err = np.where(mask, err, 0.0)
    flat = int(np.argmax(err))
    y, x, c = np.unravel_index(flat, err.shape)
    return float(err[y, x, c]), CHANNELS[c], [int(y), int(x)]


def compare(case, folder, fx):
    """One row for a case with a render in `folder`, or None when there is no render. `status` is pass/fail against Nuke."""
    from make_nuke_reference_pack import load_render
    from nuke_fidelity_cases import run_nodebased
    theirs = load_render(folder, case.id)
    if theirs is None:
        return None
    theirs = theirs.astype(np.float64)
    row = {"id": case.id, "node": case.nuke, "nodebased": case.nb_kind, "source": case.source}
    try:
        ours = run_nodebased(case)
    except Exception as exc:  # noqa: BLE001 - a case NodeBased cannot run is a finding
        row.update(status="error", note=f"{type(exc).__name__}: {str(exc)[:100]}")
        return row
    if ours is None:
        row.update(status="inexpressible", note=case.inexpressible)
        return row
    if ours.shape != theirs.shape:
        row.update(status="fail", note=f"shape {list(ours.shape)} vs Nuke {list(theirs.shape)}")
        return row
    err = np.abs(ours - theirs)
    tol = TOL + TOL * np.abs(theirs)
    bad = np.isfinite(err) & (err > tol)
    row["max_error"], row["channel"], row["worst_pixel"] = _worst(err, theirs)
    row["pixels_off"] = int(bad.any(axis=2).sum())
    row["status"] = "fail" if bad.any() else "pass"
    try:
        ref, valid = case.expected(fx)
        ref = np.asarray(ref, np.float64)
        if ref.shape == theirs.shape:
            valid = np.asarray(valid, bool) & np.isfinite(theirs)
            g_err = np.abs(theirs - ref)
            row["nuke_vs_guide"], _, row["nuke_vs_guide_pixel"] = _worst(g_err, theirs, valid)
            n_err = np.abs(ours - ref)
            row["nodebased_vs_guide"], _, _ = _worst(n_err, ref, valid)
    except Exception as exc:  # noqa: BLE001
        row["guide_note"] = f"{type(exc).__name__}"
    guide_tol = max(case.tolerance or 0.0, 1e-4)
    row["nuke_disagrees_with_guide"] = bool(row.get("nuke_vs_guide", 0.0) > guide_tol and row.get("nodebased_vs_guide", 1.0) <= guide_tol)
    return row


def lane_of(node):
    from parity_review_book import LANE_OF_GROUP, fidelity_rows, load_json, NUKE_JSON
    rows = fidelity_rows()
    group = rows.get(node, {}).get("group") or load_json(NUKE_JSON)["nodes"].get(node, {}).get("group") or "Other"
    return LANE_OF_GROUP.get(group, 11), group


def build(folder):
    from nuke_fidelity_cases import CASES, make_fixtures
    fx = make_fixtures()
    rows = []
    lanes = {}
    for case in CASES:
        row = compare(case, folder, fx)
        if row is None:
            continue
        if case.nuke not in lanes:
            lanes[case.nuke] = lane_of(case.nuke)
        row["lane"], row["group"] = lanes[case.nuke]
        for key in ("max_error", "nuke_vs_guide", "nodebased_vs_guide"):
            if key in row:
                row[key] = float(f"{row[key]:.6g}")
        rows.append(row)
    rows.sort(key=lambda r: (r["lane"], r["status"] == "pass", not r.get("nuke_disagrees_with_guide"), -(r.get("max_error") or 0), r["id"]))
    summary = {"cases": len(rows), "pass": sum(r["status"] == "pass" for r in rows),
               "nodes": len({r["node"] for r in rows}),
               "nodes_all_pass": len({r["node"] for r in rows} - {r["node"] for r in rows if r["status"] != "pass"}),
               "mismatch_by_lane": {}}
    for r in rows:
        if r["status"] != "pass":
            summary["mismatch_by_lane"][str(r["lane"])] = summary["mismatch_by_lane"].get(str(r["lane"]), 0) + 1
    return {"summary": summary, "rows": rows}


def write(data, path=OUT):
    body = ",\n".join(" " + json.dumps(r, sort_keys=True) for r in data["rows"])
    Path(path).write_text(f'{{"summary": {json.dumps(data["summary"], sort_keys=True)},\n"rows": [\n{body}\n]}}\n', encoding="utf-8")


def load(path=OUT):
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--reference-dir", required=True)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)
    data = build(args.reference_dir)
    write(data, args.out)
    print(json.dumps(data["summary"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
