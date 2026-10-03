"""Re-audit docs/PARITY_2D.md against the code, kind by kind.

For every node kind the node catalogue files under a 2D toolbar group (Image through Other) it reads the
registries and then runs the node: one small graph per kind, evaluated on the full-frame evaluator and,
where `TileExecutor.supports_tiled` says yes, on the tile path too, compared pixel for pixel; with a
mask wired, with `mix` at 0, and bypassed. It prints one line per kind:

    kind  group  registry  eval  tile  mask  mix0  bypass  docrow

and a count of the table rows by status, recomputed from the document. Run from a checkout:

    QT_QPA_PLATFORM=offscreen python tools/audit_parity_2d.py [--rows]

`--rows` also prints each table row's name and status. Nothing here edits the document: the status of a
row is a judgement the audit informs. A kind whose fixture this tool cannot build (it needs a file, a
saved document, a group) prints `skip: <reason>`, never a pass.
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from nodebased import knobs, theme  # noqa: E402
from nodebased.core import CHOICES, Dispatcher, LIMITS, NODE_LIMITS, SPECS, bypass_slot  # noqa: E402
from nodebased.imaging import Evaluator  # noqa: E402
from nodebased.nodecatalog import NODE_CATEGORIES, node_description  # noqa: E402
from nodebased.tileexec import SUPPORTED_TILED_KINDS, TileExecutor  # noqa: E402

GROUPS_2D = ("Image", "Draw", "Time", "Channel", "Color", "Filter", "Keyer", "Merge", "Transform",
             "Metadata", "Other")
DOC = ROOT / "docs" / "PARITY_2D.md"
W, H = 64, 48

# Kinds that need something this tool does not build: a file on disk, a saved document, a group.
NEEDS_FIXTURE = {
    "Read": "reads a file", "ReadBundle": "reads files", "UDIMImport": "reads files", "Precomp": "reads a saved document",
    "Group": "needs a group body", "Input": "only inside a group", "Output": "only inside a group",
    "Write": "writes a file", "Viewer": "display node", "Backdrop": "no pixels",
    "ZMerge": "needs two layered inputs (Render3D); tests.test_2d_parity_step_d1 builds them",
}


class Build:
    def __init__(self):
        self.d = Dispatcher()
        self.n = 0

    def add(self, kind, params=None, **inputs):
        self.n += 1
        key = f"n{self.n}"
        self.d.execute(dict(op="create", id=key, type=kind, params=params or {}))
        for slot, source in inputs.items():
            self.d.execute(dict(op="connect", id=key, input=slot, source=source))
        return key

    @property
    def doc(self):
        return self.d.document


def source(b):
    """A soft-edged coloured shape on a transparent plate: non-flat colour and alpha for every filter."""
    clear = b.add("Constant", dict(width=W, height=H, red=0.1, green=0.5, blue=0.2, alpha=1.0))
    box = b.add("Rectangle", dict(width=W, height=H, box_x=14.0, box_y=10.0, box_width=30.0, box_height=22.0,
                                  red=0.8, green=0.2, blue=0.1), image=clear)
    return b.add("Blur", dict(radius=2.0), image=box)


def write_identity_cube(directory):
    path = Path(directory) / "identity.cube"
    lines = ["LUT_3D_SIZE 2"]
    for blue in (0, 1):
        for green in (0, 1):
            for red in (0, 1):
                lines.append(f"{red} {green} {blue}")
    path.write_text("\n".join(lines) + "\n")
    return str(path)


def build(kind, scratch):
    """The graph under test: the kind with every required input fed `source`, plus whatever optional
    input or parameter the node cannot run without (a depth image, a matte, a vector field, a file)."""
    b = Build()
    spec = SPECS[kind]
    inputs = {slot: source(b) for slot in spec["inputs"]}
    params, extra = {}, {}
    if kind == "AppendClip":
        inputs = {"clip0": source(b)}
        extra = {"clip1": source(b)}
    elif kind in ("ZDefocus", "ZSlice"):
        extra = {"depth": b.add("Ramp", dict(width=W, height=H))}
    elif kind == "MatchGrade":
        extra = {"reference": source(b)}
    elif kind == "Inpaint":
        extra = {"matte": inputs["image"]}
    elif kind in ("VectorBlur", "STMap", "IDistort"):
        extra = {"uv": inputs["image"]}
    elif kind == "Vectorfield":
        params = {"cube_path": write_identity_cube(scratch)}
    elif kind in ("VectorToMotion", "VectorDistort", "VectorCornerPin", "GridWarpTracker"):
        vectors = b.add("SmartVector", dict(frame_start=1, frame_end=2), image=inputs["image"])
        if kind == "VectorToMotion":
            inputs = {"image": vectors}
        else:
            extra = {"vectors": vectors}
            params = {"drive": "smartvector"} if kind == "GridWarpTracker" else {}
    elif kind == "Cryptomatte":
        crypto = b.add("Encryptomatte", dict(id0="thing"), image=inputs["image"], matte0=inputs["image"])
        inputs = {"image": crypto}
        params = {"matte_list": "thing"}
    return b, b.add(kind, params, **inputs, **extra), inputs


def evaluate(doc, target):
    return Evaluator().evaluate(dict(doc, view=target))


def tiled(doc, target):
    executor = TileExecutor(evaluator=Evaluator())
    doc = dict(doc, view=target)
    if not executor.supports_tiled(doc, target):
        return None
    region = executor.canvas_region(doc, target, frame=1, tier=1)
    return executor.compose_region(doc, target, region, frame=1, tier=1).pixels


def registry(kind):
    spec = SPECS[kind]
    bad = [name for name in spec["params"] if name not in LIMITS and name not in CHOICES and name not in NODE_LIMITS.get(kind, {})
           and not isinstance(spec["params"][name], (str, list, dict))
           and name != "mix"]
    parts = []
    parts.append("laid-out" if kind in knobs.KNOB_LAYOUT else "default-panel")
    parts.append("colour" if kind in theme.COLORS else "NO-COLOUR")
    parts.append("described" if node_description(kind) else "NO-DESCRIPTION")
    if bad:
        parts.append("no-limit:" + ",".join(bad[:4]))
    return " ".join(parts)


def audit_kind(kind, scratch):
    if kind in NEEDS_FIXTURE:
        return f"skip: {NEEDS_FIXTURE[kind]}"
    spec = SPECS[kind]
    cells = {}
    try:
        b, node, inputs = build(kind, scratch)
        full = evaluate(b.doc, node)
        cells["eval"] = "ok" if np.isfinite(full).all() else "NON-FINITE"
    except Exception as exc:  # noqa: BLE001 - the audit reports, never raises
        return f"{registry(kind)} | eval ERROR {type(exc).__name__}: {str(exc)[:70]}"
    # tile path
    try:
        tiles = tiled(b.doc, node)
        if tiles is None:
            cells["tile"] = "full-frame only" if kind not in SUPPORTED_TILED_KINDS else "full-frame (config)"
        elif tiles.shape != full.shape:
            cells["tile"] = f"SHAPE {tiles.shape} vs {full.shape}"
        else:
            diff = float(np.abs(tiles - full).max())
            cells["tile"] = "matches" if diff < 1e-5 else f"DIFFERS {diff:.2g}"
    except Exception as exc:  # noqa: BLE001
        cells["tile"] = f"ERROR {type(exc).__name__}: {str(exc)[:60]}"
    # mask: an alpha-0 mask must leave the node's first image input as it was
    first = next(iter(inputs.values()), None)
    cells["mask"] = "n/a"
    same_shape = first is not None and evaluate(b.doc, first).shape == full.shape
    if not same_shape:
        cells["mask"] = cells["mix0"] = "geometry"
    if same_shape and "mask" in spec.get("optional_inputs", []) and first is not None:
        try:
            m = b.add("Constant", dict(width=W, height=H, red=0, green=0, blue=0, alpha=0.0))
            b.d.execute(dict(op="connect", id=node, input="mask", source=m))
            hidden = evaluate(b.doc, node)
            base = evaluate(b.doc, first)
            cells["mask"] = "hides" if hidden.shape == base.shape and np.allclose(hidden, base, atol=1e-5) else "DOES-NOT-HIDE"
            b.d.execute(dict(op="connect", id=node, input="mask", source=None))
        except Exception as exc:  # noqa: BLE001
            cells["mask"] = f"ERROR {str(exc)[:50]}"
    # mix 0
    cells.setdefault("mix0", "n/a")
    if same_shape and "mix" in spec["params"] and first is not None:
        try:
            b.d.execute(dict(op="set", id=node, param="mix", value=0.0))
            zero = evaluate(b.doc, node)
            base = evaluate(b.doc, first)
            cells["mix0"] = "identity" if zero.shape == base.shape and np.allclose(zero, base, atol=1e-5) else "NOT-IDENTITY"
            b.d.execute(dict(op="set", id=node, param="mix", value=1.0))
        except Exception as exc:  # noqa: BLE001
            cells["mix0"] = f"ERROR {str(exc)[:50]}"
    # bypass
    try:
        slot = bypass_slot(b.doc["nodes"][node])
        b.d.execute(dict(op="disable", id=node, value=True))
        out = evaluate(b.doc, node)
        if slot is not None and inputs.get(slot) is not None:
            base = evaluate(b.doc, inputs[slot])
            cells["bypass"] = f"passes {slot}" if out.shape == base.shape and np.allclose(out, base, atol=1e-6) else f"BYPASS-MISMATCH({slot})"
        else:
            cells["bypass"] = f"slot {slot}"
    except Exception as exc:  # noqa: BLE001
        cells["bypass"] = f"ERROR {str(exc)[:50]}"
    return (f"{registry(kind)} | eval {cells['eval']} | tile {cells['tile']} | mask {cells['mask']} | "
            f"mix0 {cells['mix0']} | bypass {cells['bypass']}")


def doc_rows():
    """[(table heading, rank, Nuke node cell, status)] for every table row of the document."""
    rows, heading = [], ""
    for line in DOC.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            heading = line[3:].strip()
        match = re.match(r"^\|\s*(\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|", line)
        if match and match.group(3) in ("supported", "partial", "missing", "n/a"):
            rows.append((heading, int(match.group(1)), match.group(2), match.group(3)))
    return rows


def row_for(kind, rows):
    for heading, rank, name, status in rows:
        if kind in [part.strip() for part in name.split("/")]:
            return f"{heading} #{rank} {status}"
    return "NO ROW"


def main():
    rows = doc_rows()
    scratch = tempfile.mkdtemp(prefix="parity-audit-")
    for group in GROUPS_2D:
        for kind in NODE_CATEGORIES[group]:
            print(f"{kind} [{group}] {audit_kind(kind, scratch)} | doc {row_for(kind, rows)}", flush=True)
    counts = {}
    for _, _, _, status in rows:
        counts[status] = counts.get(status, 0) + 1
    print("ROWS", len(rows), " ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if "--rows" in sys.argv:
        for heading, rank, name, status in rows:
            print(f"{heading} #{rank} {name}: {status}")


if __name__ == "__main__":
    main()
