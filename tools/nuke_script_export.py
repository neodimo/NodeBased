#!/usr/bin/env python3
"""Export every fidelity case as a Nuke `.nk` script, so Nuke's own pixels can become the reference.

Per case, a folder `<out>/<case id>/` holds the case's input plates as 32-bit float EXRs (the audit's own 16 x 12
fixtures, so a Nuke render compares directly with the scored case), and `<case id>.nk`: a Read per input, the Nuke
node with the case's knob values under Nuke's knob names (the case's `nuke_knobs`, written against the pairing in
docs/parity/nodebased_to_nuke.json), and a Write of `nuke_render.exr` (32-bit float, raw) beside the script.

    python tools/nuke_script_export.py --out docs/parity/review/out/nuke_scripts [--nodes Blur,Grade,...]
    # then, in Nuke: render each script's Write (see docs/parity/review/README.md), and
    python tools/nuke_script_export.py --collect docs/parity/review/out/nuke_scripts
    # copies every nuke_render.exr to docs/parity/nuke_renders/<case id>.exr, where the review book finds it.

The .nk syntax follows Nuke's saved-script format: a `version` line, then one block per node, `Name { knob value ... }`,
in stack order. A node with `inputs N` takes N items off the stack, the top one becoming input 0; `set VAR [stack 0]`
names the node on top and `push $VAR` puts it back. Values with spaces or Tcl characters are braced. Sources: the form
of the scripts Nuke itself saves (a .nk file is a Tcl script that Nuke's own `nuke.scriptOpen` / `nuke.nodePaste` read;
Nuke 17.0 Python API reference, https://learn.foundry.com/nuke/developers/17.0/pythondevguide/), the NDK Developer
Guide's account of knobs serialising themselves as text (https://learn.foundry.com/nuke/developers/17.0/ndkdevguide/),
and the Read and Write pages of the Nuke 17.0 Reference Guide for the knob names `file`, `raw`, `first`, `last`, `file_type`,
`datatype` and `compression` (https://learn.foundry.com/nuke/17.0/content/reference_guide/image_nodes/read.html,
.../image_nodes/write.html). Nobody here has opened these scripts in Nuke yet: the first person to do so should check
the first Merge render against its input plates by eye (README).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

RENDERS_DIR = ROOT / "docs" / "parity" / "nuke_renders"
MAP_JSON = ROOT / "docs" / "parity" / "nodebased_to_nuke.json"
RENDER_NAME = "nuke_render.exr"
# The script's own folder, as an expression evaluated by Nuke when the file knob is read; brackets are escaped in .nk text.
HERE = r"\[file dirname \[value root.name]]"


def tcl(value):
    """A knob value as .nk text: booleans true/false, numbers plainly, Tcl lists (already '{...}') as they are, anything
    else braced when it holds spaces or Tcl characters."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    text = str(value)
    if text.lstrip().startswith("{"):
        return text
    if text == "" or re.search(r"[\s{}\[\]$\"\\;]", text):
        return "{" + text + "}"
    return text


_PAIRS = None


def _pairs():
    global _PAIRS
    if _PAIRS is None:
        _PAIRS = json.loads(MAP_JSON.read_text(encoding="utf-8"))["pairs"]
    return _PAIRS


def knob_values(case):
    """(knobs, notes): the Nuke knob values for the case. The case's own `nuke_knobs` first; then every NodeBased param the
    case sets whose pairing is `same` (one NodeBased param, one Nuke knob, the same values) and whose Nuke knob the case did
    not state, under that Nuke name. A param paired any other way (renamed with other units, split into components) is
    left at Nuke's default and named in a comment, so nobody mistakes the script for a translation it is not."""
    knobs = list(case.nuke_knobs)
    stated = {k for k, _ in knobs}
    notes = []
    pair = _pairs().get(case.nuke, {})
    entries = pair.get("knobs", [])
    value_map = pair.get("value_map") or {}
    for param, value in (case.nb_params or {}).items():
        if param.startswith("_"):
            continue
        e = next((x for x in entries if param in (x.get("nodebased") or [])), None)
        if e is None or not e.get("nuke"):
            continue
        if set(e["nuke"]) & stated:
            continue
        if e["kind"] == "same" and len(e["nuke"]) == 1 and len(e["nodebased"]) == 1 and isinstance(value, (bool, int, float, str)):
            if isinstance(value, str):
                value = value_map.get(e["nuke"][0], {}).get(value, value)
            knobs.append((e["nuke"][0], value))
            stated.add(e["nuke"][0])
        else:
            notes.append(f"NodeBased {param}={value!r} pairs with Nuke {'/'.join(e['nuke'])} ({e['kind']}); left at Nuke's default")
    return knobs, notes


def nk_script(case):
    """The .nk text of one case."""
    inputs = list(case.nuke_inputs)
    knobs, notes = knob_values(case)
    lines = [f"# NodeBased fidelity case {case.id}: {case.nuke} ({case.source}). Written by tools/nuke_script_export.py.",
             f"# {case.statement}"] + [f"# {n}" for n in notes] + ["version 17.0", "Root {", " inputs 0", " first_frame 1", " last_frame 1", "}"]
    names = list(dict.fromkeys(inputs))
    for i, name in enumerate(names):
        lines += ["Read {", " inputs 0", f" file \"{HERE}/{name}.exr\"", " raw true", " first 1", " last 1",
                  " origfirst 1", " origlast 1", f" name Read_{name}", f" xpos {i * 110}", " ypos 0", "}",
                  f"set read_{name} [stack 0]"]
    # Push the inputs so the top of the stack is input 0: last input first.
    for name in reversed(inputs):
        lines.append(f"push $read_{name}")
    lines.append(f"{case.nuke_class} {{")
    lines.append(f" inputs {len(inputs)}")
    for knob, value in knobs:
        lines.append(f" {knob} {tcl(value)}")
    lines += [" name NodeUnderTest", " xpos 0", " ypos 120", "}",
              "Write {", f" file \"{HERE}/{RENDER_NAME}\"", " raw true", " file_type exr", " datatype {32 bit float}",
              " compression none", " first 1", " last 1", " use_limit true", " name Write_render", " xpos 0", " ypos 240", "}"]
    return "\n".join(lines) + "\n"


def pairing_names(nuke_key):
    """{nodebased param: [nuke knob names]} for one Nuke node's pairing."""
    pairs = json.loads(MAP_JSON.read_text(encoding="utf-8"))["pairs"]
    out = {}
    for e in pairs.get(nuke_key, {}).get("knobs", []):
        for p in e.get("nodebased", []) or []:
            out.setdefault(p, list(e.get("nuke", []) or []))
    return out


def input_plates(case, fixtures):
    """{input name: still image} for the case, or (None, reason) when an input is not a still plate: Time-group cases
    stamp the frame number into an animated Constant, and Roto/Tracker cases build their plate inside the document."""
    import numpy as np
    plates = {}
    for name in dict.fromkeys(case.nuke_inputs):
        if name in fixtures:
            plates[name] = fixtures[name]
            continue
        held = (case.nb_params or {}).get(f"_{name}")
        if isinstance(held, np.ndarray) and held.ndim == 3:
            plates[name] = np.asarray(held, np.float32)
            continue
        return None, f"input '{name}' is generated inside NodeBased (animated or document-built), not a still plate"
    return plates, ""


def export(out_dir, nodes=None):
    """Write one folder per case whose inputs are still plates; returns (cases written, cases skipped with a reason)."""
    from nodebased.media import write_exr
    from nuke_fidelity_cases import CASES, make_fixtures
    out_dir = Path(out_dir)
    fixtures = make_fixtures()
    count, manifest, skipped = 0, [], []
    for case in CASES:
        if nodes and case.nuke not in nodes:
            continue
        plates, reason = input_plates(case, fixtures)
        if plates is None:
            skipped.append({"id": case.id, "nuke": case.nuke, "reason": reason})
            continue
        folder = out_dir / case.id
        folder.mkdir(parents=True, exist_ok=True)
        for name, arr in plates.items():
            write_exr(folder / f"{name}.exr", arr, bits="float", compression="none")
        (folder / f"{case.id}.nk").write_text(nk_script(case), encoding="utf-8")
        manifest.append({"id": case.id, "nuke": case.nuke, "nuke_class": case.nuke_class, "script": f"{case.id}/{case.id}.nk",
                         "render": f"{case.id}/{RENDER_NAME}", "unscored": case.unscored, "expressible": case.nb_params is not None})
        count += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MANIFEST.json").write_text(json.dumps({"cases": manifest, "skipped": skipped}, indent=1) + "\n", encoding="utf-8")
    (out_dir / "render_all.py").write_text(RENDER_ALL, encoding="utf-8")
    return count, len(skipped)


RENDER_ALL = '''"""Render every case script in this folder with Nuke: `nuke -t render_all.py` (run from this folder, or pass its path).

Each script is opened, its Write_render node executed for frame 1, and the script closed. A script that fails is
listed at the end with Nuke's error. Written by tools/nuke_script_export.py.
"""
import glob
import os
import sys
import nuke

here = os.path.dirname(os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else __file__))
failed = []
for script in sorted(glob.glob(os.path.join(here, "*", "*.nk"))):
    try:
        nuke.scriptClear()
        nuke.scriptOpen(script)
        nuke.execute(nuke.toNode("Write_render"), 1, 1)
        print("rendered", os.path.basename(script))
    except Exception as exc:
        failed.append((script, str(exc)))
        print("FAILED", os.path.basename(script), exc)
nuke.scriptClear()
print(len(failed), "script(s) failed")
'''


def collect(out_dir, renders_dir=RENDERS_DIR):
    """Copy every rendered nuke_render.exr to renders_dir/<case id>.exr; returns how many were copied."""
    renders_dir = Path(renders_dir)
    renders_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for render in sorted(Path(out_dir).glob(f"*/{RENDER_NAME}")):
        shutil.copyfile(render, renders_dir / f"{render.parent.name}.exr")
        n += 1
    return n


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(ROOT / "docs" / "parity" / "review" / "out" / "nuke_scripts"))
    ap.add_argument("--nodes", help="comma-separated Nuke node names (default: every case)")
    ap.add_argument("--collect", metavar="SCRIPTS_DIR", help="copy rendered nuke_render.exr files into docs/parity/nuke_renders/")
    args = ap.parse_args(argv)
    if args.collect:
        print(f"copied {collect(args.collect)} renders into {RENDERS_DIR}")
        return 0
    n, skipped = export(args.out, set(args.nodes.split(",")) if args.nodes else None)
    print(f"wrote {n} case scripts into {args.out}; skipped {skipped} whose input is not a still plate (listed in MANIFEST.json)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
