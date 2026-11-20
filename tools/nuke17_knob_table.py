#!/usr/bin/env python3
"""Record the knob table of Nuke 17's nodes: for each node class the reference pack uses, every knob's name and kind, and the values of
its enumerations. tools/make_nuke_reference_pack.py reads the table (tools/nuke17_knobs.json) to write each knob with the Python type its
kind takes, and to mark a case not renderable when it names a knob or an enumeration value Nuke does not have.

Two halves:
    python tools/nuke17_knob_table.py --nuke /path/to/nuke-nc.sh      # host side: probes every class, rewrites tools/nuke17_knobs.json
    nuke -t tools/nuke17_knob_table.py --probe OUT.json CLASS,CLASS     # Nuke side, run by the host half, eight classes per process

Nuke Non-commercial lets a Python script reach at most 10 nodes, and deleting a node does not give its place back, so the host half
starts one Nuke process for every eight classes.
"""
import json
import os
import subprocess
import sys
import tempfile

TABLE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nuke17_knobs.json")
# Knobs that are the node's frame, not its behaviour; leaving them out keeps the table small.
UI_KNOBS = set("name help onCreate onDestroy knobChanged updateUI autolabel panel tile_color gl_color label note_font note_font_size "
               "note_font_color selected xpos ypos icon indicators hide_input rootNodeUpdated cached disable dope_sheet bookmark "
               "postage_stamp postage_stamp_frame lifetimeStart lifetimeEnd useLifetime lock_connections".split())
UI_KINDS = {"Tab_Knob", "Text_Knob", "PyScript_Knob", "Help_Knob", "Obsolete_Knob", "Script_Knob"}
BATCH = 8


def probe(out_path, classes):
    """Nuke side: create each class, read its knobs, write {class: {"knobs": {name: kind or [kind, values]}} or {"error": text}}."""
    import nuke  # noqa: F401  (only inside Nuke)
    out = {}
    for cls in classes:
        try:
            node = getattr(nuke.nodes, cls)()
            if node is None:
                raise RuntimeError("Nuke returned no node (the Non-commercial 10-node limit)")
        except Exception as exc:  # noqa: BLE001
            out[cls] = {"error": str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__}
            continue
        knobs = {}
        for name, knob in node.knobs().items():
            kind = knob.Class()
            if name in UI_KNOBS or kind in UI_KINDS:
                continue
            if kind == "Link_Knob":                     # a gizmo's knob: record the kind of the knob it links to
                try:
                    knob = knob.getLinkedKnob() or knob
                    kind = knob.Class()
                except Exception:  # noqa: BLE001
                    pass
            values = list(knob.values()) if kind == "Enumeration_Knob" and hasattr(knob, "values") else []
            knobs[name] = [kind, [v.split("\t")[0] for v in values]] if values else kind
        out[cls] = {"class": node.Class(), "knobs": knobs}
    with open(out_path, "w") as f:
        json.dump(out, f)


def build(nuke, classes):
    """Host side: probe `classes` with the Nuke executable `nuke`, eight per process, and return the merged table."""
    table = {}
    with tempfile.TemporaryDirectory() as tmp:
        out = os.path.join(tmp, "batch.json")
        for i in range(0, len(classes), BATCH):
            batch = classes[i:i + BATCH]
            subprocess.run([nuke, "-t", os.path.abspath(__file__), "--probe", out, ",".join(batch)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=300)
            with open(out) as f:
                table.update(json.load(f))
    return table


def write_table(table, path=TABLE):
    """One class per line, sorted, so a new Nuke version shows up as a readable diff."""
    lines = [json.dumps(k) + ": " + json.dumps(table[k], sort_keys=True, separators=(",", ":")) for k in sorted(table)]
    with open(path, "w") as f:
        f.write("{\n" + ",\n".join(lines) + "\n}\n")


def load_table(path=TABLE):
    with open(path) as f:
        return json.load(f)


def main(argv):
    if argv[:1] == ["--probe"]:
        probe(argv[1], [c for c in argv[2].split(",") if c])
        return 0
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--nuke", required=True, help="Nuke executable (Non-commercial: the wrapper that adds --nc)")
    args = ap.parse_args(argv)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from make_nuke_reference_pack import nuke_classes_used
    write_table(build(args.nuke, nuke_classes_used()))
    print("wrote", TABLE)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
