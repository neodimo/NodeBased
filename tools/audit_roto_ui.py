#!/usr/bin/env python3
"""Score measured NodeBased Roto/RotoPaint UI entries against the Nuke guide inventory."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SECTIONS = ("toolbar", "tool_settings", "shape_list", "viewer_editing", "timing_animation")
WEIGHTS = {"match": 1.0, "differs": 0.5, "missing": 0.0}

def score(reference, measured):
    result = {}
    total = hits = 0.0
    for section in SECTIONS:
        entries = measured[section]
        if section == "toolbar":
            entries = [tool for group in entries for tool in group["tools"]]
        section_hits = sum(WEIGHTS[e["status"]] for e in entries)
        section_score = 100.0 * section_hits / len(entries) if entries else 0.0
        result[section] = {"score": round(section_score, 1), "counts": {s: sum(e["status"] == s for e in entries) for s in WEIGHTS}, "entries": len(entries)}
        total += section_hits
        hits += len(entries)
    result["overall"] = round(100.0 * total / hits, 1) if hits else 0.0
    return result

def main():
    ref = json.loads((ROOT / "docs/parity/nuke_roto_ui.json").read_text())
    measured = json.loads((ROOT / "docs/parity/nodebased_roto_ui.json").read_text())
    print(json.dumps(score(ref, measured), indent=2))

if __name__ == "__main__":
    main()
